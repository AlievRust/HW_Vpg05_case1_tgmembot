"""Синхронный Telegram-бот-помощник с долговременной векторной памятью.

Бот использует pyTelegramBotAPI (модуль ``telebot``) и менеджер из
``chroma_manager.py``. Для каждого текстового сообщения пользователя он:

1. ищет наиболее близкие воспоминания только в памяти этого пользователя;
2. проверяет расстояние до лучшего результата;
3. просит YandexGPT определить, является ли сообщение новой, повторной или
   противоречащей информацией;
4. создаёт, пропускает или обновляет одну запись;
5. строит ответ с контекстом из нескольких похожих записей.

В документ ChromaDB записывается только исходный текст пользователя. Ответ
бота, служебный prompt и шаблонные фразы в документы памяти не попадают.

Запуск локально::

    python bot.py

Запуск в Docker::

    docker compose up --build

Основные настройки находятся в ``.env``. Секреты не следует добавлять в
образ Docker или коммитить в репозиторий.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import telebot
from dotenv import load_dotenv

from chroma_manager import ChromaManager


load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Читает целое число из окружения и возвращает default при ошибке."""

    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Некорректное значение %s, используется %s", name, default
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Читает число с плавающей точкой из окружения."""

    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Некорректное значение %s, используется %s", name, default
        )
        return default


def _model_from_env() -> tuple[str, str, str]:
    """Собирает модели чата и embeddings с поддержкой текущих имён .env.

    В существующем учебном окружении уже используются ``YANDEX_CLOUD_MODEL``
    и ``YANDEX_OPENAI_BASE_URL``. Новые более явные имена также поддерживаются,
    чтобы модуль не ломал ранее созданную конфигурацию.
    """

    folder_id = os.getenv("YANDEX_FOLDER_ID", "<folder_id>")
    chat_model = os.getenv("YANDEXGPT_MODEL") or os.getenv(
        "YANDEX_CLOUD_MODEL", f"gpt://{folder_id}/yandexgpt/latest"
    )
    # В старом .env может быть учебная заглушка replace-with-folder-id.
    if "replace-with-folder-id" in chat_model:
        chat_model = f"gpt://{folder_id}/yandexgpt/latest"

    embedding_model = os.getenv("YANDEX_EMBEDDING_MODEL")
    if not embedding_model:
        embedding_model = f"emb://{folder_id}/text-search-doc/latest"
    query_embedding_model = os.getenv("YANDEX_QUERY_EMBEDDING_MODEL")
    if not query_embedding_model:
        query_embedding_model = embedding_model.replace(
            "text-search-doc", "text-search-query"
        )
    return chat_model, embedding_model, query_embedding_model


@dataclass(frozen=True)
class MemoryDecision:
    """Решение LLM по отношению новой фразы к найденной памяти."""

    action: str
    reason: str = ""


@dataclass(frozen=True)
class MessageAnalysis:
    """Результат анализа сообщения до обращения к долговременной памяти."""

    message_type: str
    memory_worthy: bool
    normalized_fact: str = ""
    claims: tuple[dict[str, Any], ...] = ()
    reason: str = ""


class MemoryService:
    """Оркестрирует память пользователя, классификацию и ответы бота.

    Класс не знает о Telegram-сообщениях. На вход ему передаются обычные
    ``user_id`` и ``user_message``, поэтому бизнес-логику легко тестировать
    отдельно от Telegram API.

    Args:
        manager: Настроенный менеджер ChromaDB/YandexGPT.
        context_size: Сколько записей включать в контекст ответа.
        search_candidates: Сколько ближайших записей проверять при обновлении
            памяти. Для учебного проекта классифицируется лучшая запись.
        max_distance: Максимальное косинусное расстояние, при котором результат
            считается кандидатом на сравнение. Большее расстояние означает,
            что найденный текст недостаточно близок и будет создана новая запись.
    """

    ALLOWED_ACTIONS = {"new", "same", "contradiction"}
    NON_MEMORY_MESSAGE_TYPES = {"question", "command", "greeting", "other"}

    def __init__(
        self,
        manager: ChromaManager,
        *,
        context_size: int = 5,
        search_candidates: int = 5,
        max_distance: float = 0.35,
        min_confidence: float = 0.7,
    ) -> None:
        if context_size < 1 or search_candidates < 1:
            raise ValueError("Размеры контекста и списка кандидатов должны быть больше нуля")
        if max_distance < 0:
            raise ValueError("max_distance не может быть отрицательным")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence должен быть в диапазоне от 0 до 1")
        self.manager = manager
        self.context_size = context_size
        self.search_candidates = search_candidates
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.logger = logging.getLogger("memory")

    @staticmethod
    def _now() -> str:
        """Возвращает UTC-время для служебных метаданных записи."""

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _best_candidate(result: dict[str, Any]) -> dict[str, Any] | None:
        """Извлекает первый результат ChromaDB в удобный плоский словарь."""

        ids = result.get("ids") or []
        if not ids or not ids[0]:
            return None
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("cosine_distances") or result.get("distances") or [[]]
        return {
            "id": ids[0][0],
            "document": documents[0][0] if documents and documents[0] else "",
            "metadata": metadatas[0][0] if metadatas and metadatas[0] else {},
            "distance": distances[0][0] if distances and distances[0] else None,
        }

    @staticmethod
    def _local_prefilter(text: str) -> MessageAnalysis | None:
        """Отбрасывает только явно безопасные случаи без вызова LLM.

        Вопросы здесь намеренно не отбрасываются автоматически: сообщение
        может быть смешанным, например «Я живу в Екатеринбурге, а где живёшь
        ты?». В таком случае LLM должна сохранить утверждение и пропустить
        вопросительную часть.
        """

        normalized = re.sub(r"\s+", " ", text.strip()).lower()
        if normalized.startswith("/"):
            return MessageAnalysis(
                message_type="command",
                memory_worthy=False,
                reason="telegram_command",
            )

        greeting_pattern = (
            r"^(привет|здравствуйте|здравствуй|доброе утро|добрый день|"
            r"добрый вечер|спасибо|ок|окей|понятно)[!. ]*$"
        )
        if re.fullmatch(greeting_pattern, normalized, flags=re.IGNORECASE):
            return MessageAnalysis(
                message_type="greeting",
                memory_worthy=False,
                reason="obvious_greeting",
            )
        return None

    def _analyze_message(self, text: str) -> MessageAnalysis:
        """Определяет, содержит ли сообщение факты для долговременной памяти.

        LLM возвращает не только общий тип сообщения, но и массив claims. Это
        позволяет корректно обработать смешанную фразу: факт сохранить, а
        вопрос из той же фразы пропустить.
        """

        local_result = self._local_prefilter(text)
        if local_result is not None:
            return local_result

        system_prompt = (
            "Ты анализатор пользовательских сообщений для долговременной памяти. "
            "Верни только JSON без markdown. Формат: "
            '{"message_type":"question|statement|mixed|command|greeting|other",'
            '"memory_worthy":true|false,"normalized_fact":"...",'
            '"claims":[{"fact_type":"...","attribute":"...",'
            '"value":"...","normalized_fact":"...","confidence":0.0}],'
            '"reason":"..."}. '
            "В claims включай только явные утверждения пользователя о себе, "
            "его предпочтениях, целях, событиях или важных фактах. "
            "Вопросы, просьбы, приветствия и команды в claims не включай. "
            "Для смешанного сообщения извлеки утверждение, но пропусти вопрос. "
            "Фраза «Ты помнишь, что я живу в Екатеринбурге?» сама по себе "
            "является вопросом и не должна создавать новую память. "
            "normalized_fact — краткая нейтральная формулировка на русском языке."
        )
        prompt = f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{text}"

        try:
            raw = self.manager.generate(prompt, system_prompt=system_prompt)
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else raw)
            message_type = str(payload.get("message_type", "other")).strip().lower()
            raw_claims = payload.get("claims", [])
            claims: list[dict[str, Any]] = []
            if isinstance(raw_claims, list):
                for claim in raw_claims:
                    if not isinstance(claim, dict):
                        continue
                    normalized_claim = str(claim.get("normalized_fact", "")).strip()
                    if normalized_claim:
                        claims.append(
                            {
                                "fact_type": str(claim.get("fact_type", "fact")),
                                "attribute": str(claim.get("attribute", "")),
                                "value": str(claim.get("value", "")),
                                "normalized_fact": normalized_claim,
                                "confidence": float(claim.get("confidence", 0.0)),
                            }
                        )
            normalized_fact = str(payload.get("normalized_fact", "")).strip()
            if not normalized_fact and claims:
                normalized_fact = "; ".join(
                    claim["normalized_fact"] for claim in claims
                )
            memory_worthy = bool(payload.get("memory_worthy", bool(claims))) and bool(
                claims or normalized_fact
            )
            return MessageAnalysis(
                message_type=message_type,
                memory_worthy=memory_worthy,
                normalized_fact=normalized_fact,
                claims=tuple(claims),
                reason=str(payload.get("reason", "")),
            )
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, KeyError) as error:
            # При ошибке формата не сохраняем сообщение автоматически: иначе
            # вопрос может случайно попасть в долговременную память.
            self.logger.warning("message analyzer fallback: %s", error)
            return MessageAnalysis(
                message_type="other",
                memory_worthy=False,
                reason="analyzer_fallback",
            )

    def _search_user_memory(self, user_id: str, user_message: str) -> dict[str, Any]:
        """Ищет память только в пространстве конкретного Telegram-пользователя."""

        return self.manager.search_memory(
            user_message,
            user_id=user_id,
            n_results=self.search_candidates,
        )

    def _classify(self, normalized_fact: str, original_message: str, candidate: dict[str, Any]) -> MemoryDecision:
        """Просит YandexGPT классифицировать отношение двух фактов.

        Ответ модели запрашивается в JSON, но парсер дополнительно умеет
        извлекать JSON из markdown-блока. При ошибке формата безопасным
        поведением является ``new``: это может создать дубликат, но не сотрёт
        существующую память.
        """

        system_prompt = (
            "Ты классификатор памяти ассистента. Сравни новую фразу пользователя "
            "с одной записью памяти. Верни только JSON без markdown и без пояснений. "
            'Допустимый формат: {"action":"new|same|contradiction", "reason":"..."}. '
            "same означает тот же факт или перефразирование. contradiction означает, "
            "что новая фраза отрицает, заменяет или явно несовместима со старой. "
            "new означает самостоятельную новую информацию."
        )
        prompt = (
            "СТАРЫЙ НОРМАЛИЗОВАННЫЙ ФАКТ:\n"
            f"{candidate.get('metadata', {}).get('normalized_fact', candidate['document'])}\n\n"
            "СТАРАЯ ИСХОДНАЯ ФРАЗА:\n"
            f"{candidate['document']}\n\n"
            "НОВЫЙ НОРМАЛИЗОВАННЫЙ ФАКТ:\n"
            f"{normalized_fact}\n\n"
            "НОВАЯ ИСХОДНАЯ ФРАЗА ПОЛЬЗОВАТЕЛЯ:\n"
            f"{original_message}"
        )
        try:
            raw = self.manager.generate(prompt, system_prompt=system_prompt)
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else raw)
            action = str(payload.get("action", "new")).strip().lower()
            if action not in self.ALLOWED_ACTIONS:
                action = "new"
            return MemoryDecision(action=action, reason=str(payload.get("reason", "")))
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError) as error:
            self.logger.warning("memory classifier fallback: %s", error)
            return MemoryDecision(action="new", reason="classifier_fallback")

    @staticmethod
    def _memory_metadata(
        user_id: str,
        analysis: MessageAnalysis,
        *,
        updated: bool = False,
    ) -> dict[str, str | int | float | bool]:
        """Формирует плоские metadata, совместимые с ChromaDB."""

        metadata: dict[str, str | int | float | bool] = {
            "user_id": user_id,
            "memory_type": "fact",
            "message_type": analysis.message_type,
            "normalized_fact": analysis.normalized_fact,
            "claims_json": json.dumps(analysis.claims, ensure_ascii=False),
        }
        if updated:
            metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        return metadata

    def remember_message(self, user_id: str | int, user_message: str) -> str:
        """Создаёт, пропускает или обновляет память и возвращает действие.

        В документ ChromaDB передаётся ровно ``user_message``. Служебные данные
        находятся только в metadata и не становятся частью семантического
        текста памяти.
        """

        normalized_user_id = str(user_id)
        text = user_message.strip()
        if not text:
            raise ValueError("Пустое сообщение нельзя сохранить в память")

        analysis = self._analyze_message(text)
        if not analysis.memory_worthy:
            self.logger.info(
                "action: skipped user_id=%s message_type=%s reason=%s",
                normalized_user_id,
                analysis.message_type,
                analysis.reason,
            )
            return "skipped"

        confidences = [
            float(claim.get("confidence", 0.0)) for claim in analysis.claims
        ]
        if not confidences or max(confidences) < self.min_confidence:
            self.logger.info(
                "action: skipped user_id=%s message_type=%s reason=low_confidence",
                normalized_user_id,
                analysis.message_type,
            )
            return "skipped"

        # Для поиска используем нормализованный факт, а в document сохраняем
        # исходное сообщение целиком. Это помогает сопоставлять разные формы
        # одного факта и одновременно сохраняет первоисточник.
        search_text = analysis.normalized_fact or text
        result = self._search_user_memory(normalized_user_id, search_text)
        candidate = self._best_candidate(result)
        if candidate is None or candidate["distance"] is None or candidate["distance"] > self.max_distance:
            record_id = self.manager.add(
                text,
                metadatas=self._memory_metadata(normalized_user_id, analysis),
            )[0]
            self.logger.info(
                "action: created user_id=%s record_id=%s distance=%s fact=%s",
                normalized_user_id,
                record_id,
                candidate["distance"] if candidate else None,
                analysis.normalized_fact,
            )
            return "created"

        decision = self._classify(analysis.normalized_fact, text, candidate)
        if decision.action == "same":
            self.logger.info(
                "action: skipped user_id=%s record_id=%s distance=%s reason=%s",
                normalized_user_id,
                candidate["id"],
                candidate["distance"],
                decision.reason,
            )
            return "skipped"

        if decision.action == "contradiction":
            # Передаём новый оригинальный текст как document. Старый bot_response
            # или prompt здесь отсутствуют, потому что они никогда не хранились.
            metadata = dict(candidate.get("metadata") or {})
            metadata.update(self._memory_metadata(normalized_user_id, analysis, updated=True))
            self.manager.update(candidate["id"], document=text, metadata=metadata)
            self.logger.info(
                "action: updated user_id=%s record_id=%s distance=%s fact=%s reason=%s",
                normalized_user_id,
                candidate["id"],
                candidate["distance"],
                analysis.normalized_fact,
                decision.reason,
            )
            return "updated"

        record_id = self.manager.add(
            text,
            metadatas=self._memory_metadata(normalized_user_id, analysis),
        )[0]
        self.logger.info(
            "action: created user_id=%s record_id=%s distance=%s fact=%s reason=%s",
            normalized_user_id,
            record_id,
            candidate["distance"],
            analysis.normalized_fact,
            decision.reason,
        )
        return "created"

    def clear_user_memory(self, user_id: str | int) -> int:
        """Удаляет все записи ChromaDB, принадлежащие одному пользователю."""

        normalized_user_id = str(user_id)
        records = self.manager.get(
            where={"user_id": normalized_user_id},
            include=[],
        )
        ids = list(records.get("ids", []))
        if ids:
            self.manager.delete(ids=ids)
        self.logger.info(
            "action: cleared user_id=%s records=%s",
            normalized_user_id,
            len(ids),
        )
        return len(ids)

    def answer(self, user_id: str | int, user_message: str) -> str:
        """Генерирует ответ на основе контекста из похожих пользовательских фраз."""

        system_prompt = os.getenv(
            "BOT_SYSTEM_PROMPT",
            "Ты полезный Telegram-ассистент. Отвечай на русском языке кратко и по делу. "
            "Используй контекст памяти только если он относится к вопросу пользователя. "
            "Не упоминай техническую реализацию памяти.",
        )
        return self.manager.answer_with_memory(
            user_message,
            user_id=str(user_id),
            n_results=self.context_size,
            system_prompt=system_prompt,
        )


def build_manager() -> ChromaManager:
    """Создаёт ChromaManager из переменных окружения проекта."""

    chat_model, embedding_model, query_embedding_model = _model_from_env()
    return ChromaManager(
        collection_name=os.getenv("CHROMA_COLLECTION", "assistant_memory"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY")
        or os.getenv("CHROMA_DIR", "data/chroma"),
        api_key=os.getenv("YANDEX_API_KEY"),
        base_url=os.getenv("YANDEX_BASE_URL")
        or os.getenv("YANDEX_OPENAI_BASE_URL", ChromaManager.DEFAULT_BASE_URL),
        chat_model=chat_model,
        embedding_model=embedding_model,
        query_embedding_model=query_embedding_model,
    )


def build_bot() -> tuple[telebot.TeleBot, MemoryService]:
    """Создаёт Telegram-бота и сервис памяти, не запуская polling."""

    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN или BOT_TOKEN")

    manager = build_manager()
    service = MemoryService(
        manager,
        context_size=_env_int("MEMORY_CONTEXT_SIZE", _env_int("TOP_K", 5)),
        search_candidates=_env_int("MEMORY_SEARCH_CANDIDATES", 5),
        max_distance=_env_float("MEMORY_MAX_DISTANCE", 0.35),
        min_confidence=_env_float("MEMORY_MIN_CONFIDENCE", 0.7),
    )
    bot = telebot.TeleBot(token, threaded=False)
    pending_clear: set[str] = set()

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: Any) -> None:
        """Отвечает на стартовую и справочную команды."""

        bot.reply_to(
            message,
            "Привет! Я помощник с памятью. Напишите сообщение, и я постараюсь "
            "учесть его в дальнейшем диалоге.\n\n"
            "Для удаления всей вашей памяти используйте /forget_me.",
        )

    @bot.message_handler(commands=["forget_me", "clear_memory"])
    def request_memory_clear(message: Any) -> None:
        """Запрашивает явное подтверждение перед удалением памяти пользователя."""

        user_id = str(message.from_user.id)
        pending_clear.add(user_id)
        bot.reply_to(
            message,
            "Вся сохранённая память этого пользователя будет удалена. "
            "Если вы уверены, отправьте /forget_me_confirm.",
        )

    @bot.message_handler(commands=["forget_me_confirm"])
    def confirm_memory_clear(message: Any) -> None:
        """Удаляет память пользователя после отдельного подтверждения."""

        user_id = str(message.from_user.id)
        if user_id not in pending_clear:
            bot.reply_to(message, "Сначала отправьте /forget_me.")
            return
        pending_clear.discard(user_id)
        deleted = service.clear_user_memory(user_id)
        bot.reply_to(message, f"Готово. Удалено записей памяти: {deleted}.")

    @bot.message_handler(content_types=["text"])
    def handle_text(message: Any) -> None:
        """Обрабатывает обычное текстовое сообщение пользователя."""

        user_message = (message.text or "").strip()
        if not user_message:
            return
        user_id = str(message.from_user.id)
        try:
            service.remember_message(user_id, user_message)
            response = service.answer(user_id, user_message)
            bot.reply_to(message, response)
        except Exception:
            # Пользователю не показываем traceback или содержимое ключей, но
            # сохраняем stack trace в консоли контейнера для диагностики.
            logging.getLogger(__name__).exception(
                "Ошибка обработки сообщения user_id=%s", user_id
            )
            bot.reply_to(message, "Извините, при обработке сообщения произошла ошибка.")

    return bot, service


def main() -> None:
    """Настраивает логирование и запускает синхронный long polling."""

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot, _service = build_bot()
    logging.getLogger(__name__).info("Telegram bot started")
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)


if __name__ == "__main__":
    main()
