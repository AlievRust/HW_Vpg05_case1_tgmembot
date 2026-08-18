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


def _model_from_env() -> tuple[str, str]:
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
    return chat_model, embedding_model


@dataclass(frozen=True)
class MemoryDecision:
    """Решение LLM по отношению новой фразы к найденной памяти."""

    action: str
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

    def __init__(
        self,
        manager: ChromaManager,
        *,
        context_size: int = 5,
        search_candidates: int = 5,
        max_distance: float = 0.35,
    ) -> None:
        if context_size < 1 or search_candidates < 1:
            raise ValueError("Размеры контекста и списка кандидатов должны быть больше нуля")
        if max_distance < 0:
            raise ValueError("max_distance не может быть отрицательным")
        self.manager = manager
        self.context_size = context_size
        self.search_candidates = search_candidates
        self.max_distance = max_distance
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

    def _search_user_memory(self, user_id: str, user_message: str) -> dict[str, Any]:
        """Ищет память только в пространстве конкретного Telegram-пользователя."""

        return self.manager.search_memory(
            user_message,
            user_id=user_id,
            n_results=self.search_candidates,
        )

    def _classify(self, user_message: str, candidate: dict[str, Any]) -> MemoryDecision:
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
            "СТАРАЯ ЗАПИСЬ ПАМЯТИ:\n"
            f"{candidate['document']}\n\n"
            "НОВАЯ ФРАЗА ПОЛЬЗОВАТЕЛЯ:\n"
            f"{user_message}"
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

        result = self._search_user_memory(normalized_user_id, text)
        candidate = self._best_candidate(result)
        if candidate is None or candidate["distance"] is None or candidate["distance"] > self.max_distance:
            record_id = self.manager.add(
                text,
                metadatas={
                    "user_id": normalized_user_id,
                    "created_at": self._now(),
                },
            )[0]
            self.logger.info(
                "action: created user_id=%s record_id=%s distance=%s",
                normalized_user_id,
                record_id,
                candidate["distance"] if candidate else None,
            )
            return "created"

        decision = self._classify(text, candidate)
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
            metadata.update(
                {
                    "user_id": normalized_user_id,
                    "updated_at": self._now(),
                }
            )
            self.manager.update(candidate["id"], document=text, metadata=metadata)
            self.logger.info(
                "action: updated user_id=%s record_id=%s distance=%s reason=%s",
                normalized_user_id,
                candidate["id"],
                candidate["distance"],
                decision.reason,
            )
            return "updated"

        record_id = self.manager.add(
            text,
            metadatas={
                "user_id": normalized_user_id,
                "created_at": self._now(),
            },
        )[0]
        self.logger.info(
            "action: created user_id=%s record_id=%s distance=%s reason=%s",
            normalized_user_id,
            record_id,
            candidate["distance"],
            decision.reason,
        )
        return "created"

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

    chat_model, embedding_model = _model_from_env()
    return ChromaManager(
        collection_name=os.getenv("CHROMA_COLLECTION", "assistant_memory"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY")
        or os.getenv("CHROMA_DIR", "data/chroma"),
        api_key=os.getenv("YANDEX_API_KEY"),
        base_url=os.getenv("YANDEX_BASE_URL")
        or os.getenv("YANDEX_OPENAI_BASE_URL", ChromaManager.DEFAULT_BASE_URL),
        chat_model=chat_model,
        embedding_model=embedding_model,
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
    )
    bot = telebot.TeleBot(token, threaded=False)

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: Any) -> None:
        """Отвечает на стартовую и справочную команды."""

        bot.reply_to(
            message,
            "Привет! Я помощник с памятью. Напишите сообщение, и я постараюсь "
            "учесть его в дальнейшем диалоге.",
        )

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
