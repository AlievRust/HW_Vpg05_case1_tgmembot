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
from collections import defaultdict, deque
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
            считается кандидатом для проверки LLM. Большее расстояние означает,
            что найденный текст недостаточно близок и будет создана новая запись.
    """

    ALLOWED_ACTIONS = {"new", "same", "contradiction"}
    MEMORY_KINDS = {"fact", "state", "instruction", "task", "note"}
    NON_MEMORY_MESSAGE_TYPES = {"question", "command", "greeting", "other"}

    def __init__(
        self,
        manager: ChromaManager,
        *,
        context_size: int = 5,
        search_candidates: int = 5,
        max_distance: float = 0.8,
        min_confidence: float = 0.7,
        recent_turns: int = 3,
    ) -> None:
        if context_size < 1 or search_candidates < 1 or recent_turns < 1:
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
        self.recent_turns = recent_turns
        self.dialogue_history: dict[str, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=recent_turns)
        )
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

    def _recent_dialogue(self, user_id: str) -> str:
        """Форматирует несколько последних реплик для разрешения контекста."""

        turns = self.dialogue_history.get(user_id)
        if not turns:
            return "Контекст предыдущих реплик отсутствует."
        lines: list[str] = []
        for user_text, assistant_text in turns:
            lines.append(f"Пользователь: {user_text}")
            lines.append(f"Ассистент: {assistant_text}")
        return "\n".join(lines)

    def _analyze_message(self, user_id: str, text: str) -> MessageAnalysis:
        """Определяет, содержит ли сообщение факты для долговременной памяти.

        LLM возвращает не только общий тип сообщения, но и массив claims. Это
        позволяет корректно обработать смешанную фразу: факт сохранить, а
        вопрос из той же фразы пропустить.
        """

        local_result = self._local_prefilter(text)
        if local_result is not None:
            return local_result

        system_prompt = (
            "Ты анализатор сообщений для универсальной долговременной памяти ассистента. "
            "Верни только JSON без markdown. Формат: "
            '{"message_type":"question|statement|mixed|command|greeting|other",'
            '"memory_worthy":true|false,"normalized_fact":"...",'
            '"claims":[{"memory_kind":"fact|state|instruction|task|note",'
            '"subject":"...","predicate":"...","value":"...",'
            '"normalized_fact":"...","confidence":0.0}],'
            '"reason":"..."}. '
            "В claims включай только информацию, которую разумно сохранить для будущего диалога. "
            "fact — устойчивый факт или справочная информация; state — актуальное, "
            "потенциально изменяемое состояние пользователя или другой сущности (место проживания, "
            "цена, статус); instruction — явное указание, как ассистент должен отвечать или вести себя; "
            "task — намерение, задача или напоминание; note — явная свободная заметка, которую нельзя "
            "надёжно разложить на другой тип. Для каждого claim заполни subject, predicate и value. "
            "Например, «билет Екатеринбург—Берлин стоит 30 000 рублей» — state с subject «билет Екатеринбург—Берлин», "
            "predicate «стоимость», value «30000 RUB»; «веди себя как Гомер Симпсон» — instruction. "
            "Не сохраняй обычные вопросы, просьбы о разовом действии, приветствия и команды. "
            "Для смешанного сообщения извлеки утверждение или инструкцию, но пропусти вопрос. "
            "Контекст предыдущих реплик используй только для разрешения неполных фраз и местоимений. "
            "Например, если после разговора о цене пиццы пользователь говорит «у нас 900 рублей», "
            "сохрани «Пицца в названном городе стоит 900 рублей», а не бюджет пользователя. "
            "Фраза «Ты помнишь, что я живу в Екатеринбурге?» сама по себе "
            "является вопросом и не должна создавать новую память. "
            "normalized_fact — краткая нейтральная формулировка на русском языке."
        )
        prompt = (
            f"КРАТКИЙ КОНТЕКСТ ДИАЛОГА:\n{self._recent_dialogue(user_id)}\n\n"
            f"ТЕКУЩЕЕ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{text}"
        )

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
                        memory_kind = str(claim.get("memory_kind", "fact")).strip().lower()
                        if memory_kind not in self.MEMORY_KINDS:
                            memory_kind = "fact"
                        claims.append(
                            {
                                "memory_kind": memory_kind,
                                "subject": str(claim.get("subject", "")),
                                "predicate": str(claim.get("predicate", "")),
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

        now = datetime.now(timezone.utc).isoformat()
        primary_claim = analysis.claims[0] if analysis.claims else {}
        memory_kind = str(primary_claim.get("memory_kind", "fact"))
        metadata: dict[str, str | int | float | bool] = {
            "user_id": user_id,
            "memory_type": memory_kind,
            "message_type": analysis.message_type,
            "normalized_fact": analysis.normalized_fact,
            "semantic_text": analysis.normalized_fact,
            "claims_json": json.dumps(analysis.claims, ensure_ascii=False),
            "status": "active",
            "updated_at": now,
        }
        if not updated:
            metadata["created_at"] = now
        # Дублируем основные поля claim в плоских metadata. Это даёт
        # приложению явную структуру записи: например, state/user/residence/Berlin.
        # Исходный текст при этом по-прежнему остаётся единственным document.
        for field in ("subject", "predicate", "value"):
            value = str(primary_claim.get(field, "")).strip()
            if value:
                metadata[field] = value
        # Для меняющегося состояния это начало актуальности именно нового значения.
        if memory_kind == "state":
            metadata["valid_from"] = now
            metadata["valid_to"] = ""
        return metadata

    @staticmethod
    def _analysis_for_claim(
        analysis: MessageAnalysis, claim: dict[str, Any]
    ) -> MessageAnalysis:
        """Возвращает анализ, содержащий ровно одну атомарную запись памяти."""

        return MessageAnalysis(
            message_type=analysis.message_type,
            memory_worthy=True,
            normalized_fact=str(claim["normalized_fact"]),
            claims=(claim,),
            reason=analysis.reason,
        )

    def _search_memory_for_claim(
        self, user_id: str, claim: dict[str, Any]
    ) -> dict[str, Any]:
        """Ищет кандидата в подходящем типе памяти.

        У старых записей ещё стоит ``memory_type=fact``. Поэтому для ``state``
        поиск включает и ``state``, и старый ``fact``: это сохраняет прежнее
        обновление места проживания и других пользовательских состояний, но
        не сравнивает их с инструкциями или задачами.
        """

        memory_kind = str(claim.get("memory_kind", "fact"))
        if memory_kind == "state":
            return self.manager.search(
                str(claim["normalized_fact"]),
                n_results=self.search_candidates,
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"$or": [{"memory_type": "state"}, {"memory_type": "fact"}]},
                    ]
                },
            )
        return self.manager.search_memory(
            str(claim["normalized_fact"]),
            user_id=user_id,
            memory_type=memory_kind,
            n_results=self.search_candidates,
        )

    def _remember_claim(
        self,
        user_id: str,
        source_text: str,
        analysis: MessageAnalysis,
        claim: dict[str, Any],
    ) -> str:
        """Сохраняет одну атомарную запись и возвращает выполненное действие."""

        claim_analysis = self._analysis_for_claim(analysis, claim)
        semantic_embedding = self.manager.embed_documents(
            claim_analysis.normalized_fact
        )[0]
        result = self._search_memory_for_claim(user_id, claim)
        candidate = self._best_candidate(result)
        if (
            candidate is None
            or candidate["distance"] is None
            or candidate["distance"] > self.max_distance
        ):
            record_id = self.manager.add(
                source_text,
                metadatas=self._memory_metadata(user_id, claim_analysis),
                embeddings=[semantic_embedding],
            )[0]
            self.logger.info(
                "action: created user_id=%s record_id=%s kind=%s distance=%s fact=%s",
                user_id,
                record_id,
                claim["memory_kind"],
                candidate["distance"] if candidate else None,
                claim["normalized_fact"],
            )
            return "created"

        decision = self._classify(claim_analysis.normalized_fact, source_text, candidate)
        if decision.action == "same":
            self.logger.info(
                "action: skipped user_id=%s record_id=%s kind=%s distance=%s reason=%s",
                user_id,
                candidate["id"],
                claim["memory_kind"],
                candidate["distance"],
                decision.reason,
            )
            return "skipped"

        # Сохраняем прежнее поведение обновления только для актуальных состояний
        # и взаимоисключающих инструкций. Остальные противоречивые сведения
        # остаются самостоятельными заметками с собственной датой.
        if decision.action == "contradiction" and claim["memory_kind"] in {"state", "instruction"}:
            metadata = dict(candidate.get("metadata") or {})
            metadata.update(self._memory_metadata(user_id, claim_analysis, updated=True))
            self.manager.update(
                candidate["id"],
                document=source_text,
                metadata=metadata,
                embedding=semantic_embedding,
            )
            self.logger.info(
                "action: updated user_id=%s record_id=%s kind=%s distance=%s fact=%s reason=%s",
                user_id,
                candidate["id"],
                claim["memory_kind"],
                candidate["distance"],
                claim["normalized_fact"],
                decision.reason,
            )
            return "updated"

        record_id = self.manager.add(
            source_text,
            metadatas=self._memory_metadata(user_id, claim_analysis),
            embeddings=[semantic_embedding],
        )[0]
        self.logger.info(
            "action: created user_id=%s record_id=%s kind=%s distance=%s fact=%s reason=%s",
            user_id,
            record_id,
            claim["memory_kind"],
            candidate["distance"],
            claim["normalized_fact"],
            decision.reason,
        )
        return "created"

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

        analysis = self._analyze_message(normalized_user_id, text)
        if not analysis.memory_worthy:
            self.logger.info(
                "action: skipped user_id=%s message_type=%s reason=%s",
                normalized_user_id,
                analysis.message_type,
                analysis.reason,
            )
            return "skipped"

        accepted_claims = [
            claim
            for claim in analysis.claims
            if float(claim.get("confidence", 0.0)) >= self.min_confidence
        ]
        if not accepted_claims:
            self.logger.info(
                "action: skipped user_id=%s message_type=%s reason=low_confidence",
                normalized_user_id,
                analysis.message_type,
            )
            return "skipped"

        actions = [
            self._remember_claim(normalized_user_id, text, analysis, claim)
            for claim in accepted_claims
        ]
        # Сохраняем совместимый с прежним кодом один итоговый статус сообщения.
        if "updated" in actions:
            return "updated"
        if "created" in actions:
            return "created"
        return "skipped"

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
        self.dialogue_history.pop(normalized_user_id, None)
        self.logger.info(
            "action: cleared user_id=%s records=%s",
            normalized_user_id,
            len(ids),
        )
        return len(ids)

    def _active_instructions(self, user_id: str | int) -> list[str]:
        """Возвращает актуальные инструкции пользователя в порядке свежести."""

        records = self.manager.get(
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"memory_type": "instruction"},
                ]
            },
            include=["documents", "metadatas"],
        )
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        instructions: list[tuple[str, str]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if metadata.get("status", "active") != "active":
                continue
            instruction = str(metadata.get("normalized_fact") or document).strip()
            if instruction:
                instructions.append((str(metadata.get("updated_at", "")), instruction))
        return [instruction for _, instruction in sorted(instructions, reverse=True)]

    def answer(self, user_id: str | int, user_message: str) -> str:
        """Генерирует ответ с релевантной памятью и активными инструкциями."""

        system_prompt = os.getenv(
            "BOT_SYSTEM_PROMPT",
            "Ты полезный Telegram-ассистент. Отвечай на русском языке кратко и по делу. "
            "Используй контекст памяти только если он относится к вопросу пользователя. "
            "Не упоминай техническую реализацию памяти.",
        )
        instructions = self._active_instructions(user_id)
        if instructions:
            instruction_block = "\n".join(f"- {item}" for item in instructions)
            system_prompt = (
                f"{system_prompt}\n\n"
                "Активные пользовательские инструкции для этого диалога "
                "(выполняй их, если они не конфликтуют с безопасностью и системными правилами):\n"
                f"{instruction_block}"
            )
        response = self.manager.answer_with_memory(
            user_message,
            user_id=str(user_id),
            n_results=self.context_size,
            system_prompt=system_prompt,
        )
        self.dialogue_history[str(user_id)].append((user_message, response))
        return response


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
        # Порог используется для допуска кандидата к LLM-классификатору.
        # Значение 0.8 позволяет распознавать противоречия вроде
        # «Екатеринбург» -> «Берлин», даже если их embedding-расстояние выше
        # порога строгого совпадения.
        max_distance=_env_float("MEMORY_MAX_DISTANCE", 0.8),
        min_confidence=_env_float("MEMORY_MIN_CONFIDENCE", 0.7),
        recent_turns=_env_int("MEMORY_RECENT_TURNS", 3),
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
