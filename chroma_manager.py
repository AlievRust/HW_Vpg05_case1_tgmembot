"""Учебный менеджер векторной памяти на базе ChromaDB и YandexGPT.

Модуль intentionally не привязан к Telegram или к конкретному боту. Его задача —
дать приложению понятный слой для работы с долговременной памятью:

* создать или открыть коллекцию ChromaDB;
* добавить, обновить, получить и удалить записи;
* искать записи по смыслу, идентификатору, метаданным и точному фрагменту текста;
* полностью очистить коллекцию;
* передать найденный контекст в YandexGPT через OpenAI-совместимый API.

Для генерации ответов используется ``ChatOpenAI``, а для embeddings —
OpenAI-совместимый клиент с прямой передачей текста в Yandex AI Studio. Прямая
передача нужна потому, что endpoint Yandex ожидает текстовый ``input`` и не
принимает массивы токенов, которые некоторые версии LangChain формируют
автоматически.

Минимальная настройка через файл ``.env``::

    YANDEX_API_KEY=ваш_api_ключ
    YANDEX_BASE_URL=https://ai.api.cloud.yandex.net/v1
    YANDEX_FOLDER_ID=ваш_id_каталога
    YANDEXGPT_MODEL=gpt://ваш_id_каталога/yandexgpt/latest
    YANDEX_EMBEDDING_MODEL=emb://ваш_id_каталога/text-search-doc/latest

Зависимости перечислены в ``requirements.txt``. Важно: записи документов и
поисковые запросы должны векторизоваться одной совместимой парой моделей. Если
модель эмбеддингов меняется для уже существующей коллекции, старую коллекцию
лучше пересоздать, иначе размерность или смысл векторов может не совпасть.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Загружаем настройки из ближайшего .env, если он существует. Уже заданные
# переменные окружения dotenv не перезаписывает, что удобно для production.
load_dotenv()


Metadata = Mapping[str, str | int | float | bool]
"""Допустимый набор простых типов метаданных ChromaDB."""


class _LangChainEmbeddingAdapter:
    """Адаптирует ``OpenAIEmbeddings`` LangChain к интерфейсу ChromaDB.

    Низкоуровневая коллекция ChromaDB вызывает embedding-функцию как обычную
    функцию: ``embedding_function(["текст 1", "текст 2"])``. В LangChain
    используется другой интерфейс — ``embed_documents`` и ``embed_query``.
    Адаптер скрывает это различие и оставляет в основном классе единый способ
    подключения эмбеддингов.

    Отдельный метод ``embed_query`` нужен для прямого использования адаптера в
    отладочном коде; сама коллекция ChromaDB для ``query_texts`` вызывает
    ``__call__``.
    """

    def __init__(self, embeddings: OpenAIEmbeddings) -> None:
        self._embeddings = embeddings

    @staticmethod
    def name() -> str:
        """Возвращает имя embedding-функции в формате ChromaDB 1.5+."""

        return "yandex-openai-compatible"

    def get_config(self) -> dict[str, str]:
        """Возвращает минимальную конфигурацию для протокола ChromaDB."""

        return {"provider": "langchain-openai"}

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Векторизует пакет документов в формате, который ждёт ChromaDB."""

        return [list(vector) for vector in self._embeddings.embed_documents(list(input))]

    def embed_query(self, input: str | Sequence[str]) -> list[list[float]]:
        """Векторизует запросы в формате ``[vector, ...]`` ChromaDB."""

        texts = [input] if isinstance(input, str) else list(input)
        return [list(self._embeddings.embed_query(text)) for text in texts]


class _YandexOpenAIEmbeddingAdapter:
    """Получает embeddings напрямую через OpenAI-совместимый Yandex API.

    ``OpenAIEmbeddings`` LangChain по умолчанию токенизирует текст локально и
    может отправить в endpoint массив числовых token IDs. OpenAI API это обычно
    допускает, но Yandex AI Studio в этом режиме ожидает строковый ``input``.
    Поэтому здесь каждый исходный текст отправляется напрямую как строка.

    Для документов и поисковых запросов используются разные модели Yandex:
    ``text-search-doc`` и ``text-search-query``. Это соответствует назначению
    моделей: одна предназначена для больших текстов, вторая — для коротких
    поисковых запросов.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        document_model: str,
        query_model: str,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.document_model = document_model
        self.query_model = query_model

    @staticmethod
    def name() -> str:
        """Возвращает имя embedding-функции в формате ChromaDB 1.5+."""

        return "yandex-openai-compatible"

    def get_config(self) -> dict[str, str]:
        """Возвращает конфигурацию моделей для протокола ChromaDB."""

        return {
            "provider": "yandex-openai-compatible",
            "document_model": self.document_model,
            "query_model": self.query_model,
        }

    def _embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        """Отправляет тексты по одному, сохраняя строковый формат input."""

        vectors: list[list[float]] = []
        for text in texts:
            response = self._client.embeddings.create(
                input=text,
                model=model,
                encoding_format="float",
            )
            vectors.append(list(response.data[0].embedding))
        return vectors

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Создаёт embeddings документов с помощью document-модели."""

        return self._embed(list(input), self.document_model)

    def embed_query(self, input: str | Sequence[str]) -> list[list[float]]:
        """Создаёт embeddings запросов с помощью query-модели."""

        texts = [input] if isinstance(input, str) else list(input)
        return self._embed(texts, self.query_model)


class _CallableEmbeddingAdapter:
    """Добавляет служебные методы ChromaDB к простой callable-функции.

    Пользовательская функция может быть минимальной и реализовывать только
    ``__call__``. ChromaDB 1.5 дополнительно проверяет ``name()``, поэтому
    менеджер автоматически оборачивает такие функции этим адаптером.
    """

    def __init__(self, function: Any) -> None:
        self._function = function

    @staticmethod
    def name() -> str:
        """Возвращает стабильное имя пользовательского embedding-провайдера."""

        return "custom-callable"

    def get_config(self) -> dict[str, str]:
        """Возвращает пустую конфигурацию: функцию нельзя сериализовать автоматически."""

        return {}

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Передаёт тексты пользовательской функции и нормализует её результат."""

        return [list(vector) for vector in self._function(input)]

    def embed_query(self, input: str | Sequence[str]) -> list[list[float]]:
        """Векторизует поисковые запросы через ту же callable-функцию."""

        texts = [input] if isinstance(input, str) else list(input)
        return self(texts)


class ChromaManager:
    """Управляет одной коллекцией ChromaDB и подключением к YandexGPT.

    По умолчанию создаётся локальная персистентная база в каталоге
    ``data/chroma``. Для тестов можно передать ``persist_directory=None`` — в
    таком режиме будет создана временная in-memory база. Для подключения к
    удалённому Chroma-серверу можно использовать параметры ``host`` и ``port``.

    Параметры подключения к OpenAI-совместимому API можно передать явно или
    задать переменными окружения:

    ``YANDEX_API_KEY`` / ``OPENAI_API_KEY``
        API-ключ. Предпочтительно использовать имя ``YANDEX_API_KEY``.
    ``YANDEX_BASE_URL``
        Base URL API, по умолчанию ``https://ai.api.cloud.yandex.net/v1``.
    ``YANDEX_FOLDER_ID``
        ID каталога. Используется только для удобного построения моделей по
        умолчанию.
    ``YANDEXGPT_MODEL``
        Модель чата, например ``gpt://<folder_id>/yandexgpt/latest``.
    ``YANDEX_EMBEDDING_MODEL``
        Модель эмбеддингов, например
        ``emb://<folder_id>/text-search-doc/latest``.
    ``YANDEX_QUERY_EMBEDDING_MODEL``
        Модель для коротких поисковых запросов, например
        ``emb://<folder_id>/text-search-query/latest``.

    Args:
        collection_name: Имя коллекции ChromaDB.
        persist_directory: Путь к локальному хранилищу. ``None`` включает
            временную in-memory базу.
        api_key: Ключ OpenAI-совместимого API.
        base_url: Base URL OpenAI-совместимого API.
        chat_model: Название модели YandexGPT для генерации ответов.
        embedding_model: Название модели для векторизации документов и запросов.
        embedding_function: Готовая функция эмбеддингов. Удобна для тестов и
            для случая, когда эмбеддинги предоставляет другой сервис.
        collection_metadata: Дополнительные настройки коллекции, например
            ``{"hnsw:space": "cosine"}``.
        host: Адрес удалённого Chroma-сервера. Если задан, используется вместо
            локального клиента.
        port: Порт удалённого Chroma-сервера.
        ssl: Использовать HTTPS при подключении к удалённому Chroma.
        headers: Дополнительные HTTP-заголовки для удалённого Chroma.

    Raises:
        ValueError: Если имя коллекции пустое или одновременно заданы
            несовместимые параметры подключения.

    Example:
        >>> manager = ChromaManager(collection_name="assistant_memory")
        >>> manager.add("Пользователь любит бег и книги", id="user-1")
        ['user-1']
        >>> manager.search("Какие у пользователя интересы?", n_results=1)
        { ... }
    """

    DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
    DEFAULT_PERSIST_DIRECTORY = "data/chroma"
    DEFAULT_COLLECTION_METADATA = {"hnsw:space": "cosine"}

    def __init__(
        self,
        collection_name: str = "assistant_memory",
        persist_directory: str | Path | None = DEFAULT_PERSIST_DIRECTORY,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        query_embedding_model: str | None = None,
        embedding_function: Any | None = None,
        collection_metadata: Mapping[str, Any] | None = None,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name не должен быть пустым")
        if host and persist_directory is not None:
            raise ValueError(
                "Для удалённого Chroma задайте persist_directory=None, "
                "либо не передавайте host."
            )

        self.collection_name = collection_name
        self.persist_directory = (
            Path(persist_directory) if persist_directory is not None else None
        )
        self.api_key = api_key or os.getenv("YANDEX_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("YANDEX_BASE_URL", self.DEFAULT_BASE_URL)

        folder_id = os.getenv("YANDEX_FOLDER_ID", "<folder_id>")
        self.chat_model_name = chat_model or os.getenv(
            "YANDEXGPT_MODEL", f"gpt://{folder_id}/yandexgpt/latest"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "YANDEX_EMBEDDING_MODEL",
            f"emb://{folder_id}/text-search-doc/latest",
        )
        self.query_embedding_model_name = query_embedding_model or os.getenv(
            "YANDEX_QUERY_EMBEDDING_MODEL",
            self.embedding_model_name.replace("text-search-doc", "text-search-query"),
        )

        # Если функция эмбеддингов передана снаружи, не создаём второй клиент.
        # Это позволяет использовать локальную mock-функцию в тестах без сети.
        # В обычном режиме LangChain-объект оборачивается адаптером под
        # низкоуровневый интерфейс ChromaDB (callable с аргументом input).
        if embedding_function is None:
            self.embedding_function = _YandexOpenAIEmbeddingAdapter(
                api_key=self.api_key,
                base_url=self.base_url,
                document_model=self.embedding_model_name,
                query_model=self.query_embedding_model_name,
            )
        elif hasattr(embedding_function, "embed_documents"):
            langchain_embeddings = embedding_function
            self.embedding_function = _LangChainEmbeddingAdapter(langchain_embeddings)
        elif callable(embedding_function) and not hasattr(embedding_function, "name"):
            self.embedding_function = _CallableEmbeddingAdapter(embedding_function)
        else:
            self.embedding_function = embedding_function

        # LLM создаётся лениво в _get_llm. Благодаря этому операции CRUD можно
        # тестировать локально и без API-ключа, а ключ понадобится только тогда,
        # когда приложение действительно попросит сгенерировать ответ.
        self.llm: ChatOpenAI | None = None

        if host:
            # HttpClient нужен, когда Chroma работает отдельным сервисом.
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=dict(headers or {}),
            )
        elif self.persist_directory is None:
            self.client = chromadb.EphemeralClient()
        else:
            # PersistentClient сам создаёт каталог базы при первом обращении.
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Косинусная метрика особенно удобна для текстовых embeddings: чем
        # меньше расстояние, тем ближе документы по смыслу. Если коллекция уже
        # существует, ChromaDB сохранит её исходную метрику.
        collection_options = dict(self.DEFAULT_COLLECTION_METADATA)
        if collection_metadata:
            collection_options.update(collection_metadata)
        self.collection_metadata = collection_options

        collection_kwargs: dict[str, Any] = {
            "name": self.collection_name,
            "embedding_function": self.embedding_function,
            "metadata": collection_options,
        }
        self.collection: Collection = self.client.get_or_create_collection(
            **collection_kwargs
        )

    # ---------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ---------------------------------------------------------------------

    @staticmethod
    def _as_list(value: str | Sequence[str], argument_name: str) -> list[str]:
        """Приводит одну строку или последовательность строк к списку.

        Строка является последовательностью в Python, поэтому отдельная
        обработка здесь обязательна: без неё текст превратился бы в список
        отдельных символов.
        """

        if isinstance(value, str):
            result = [value]
        else:
            result = list(value)
        if not result:
            raise ValueError(f"{argument_name} не должен быть пустым")
        if any(not isinstance(item, str) or not item for item in result):
            raise TypeError(f"Все элементы {argument_name} должны быть непустыми строками")
        return result

    @staticmethod
    def _normalise_metadatas(
        metadatas: Metadata | Sequence[Metadata] | None,
        size: int,
    ) -> list[dict[str, str | int | float | bool]] | None:
        """Проверяет и разворачивает метаданные в формат, ожидаемый ChromaDB."""

        if metadatas is None:
            return None
        if isinstance(metadatas, Mapping):
            values: list[Metadata] = [metadatas] * size
        else:
            values = list(metadatas)
            if len(values) != size:
                raise ValueError("Количество metadatas должно совпадать с количеством записей")

        allowed = (str, int, float, bool)
        result: list[dict[str, str | int | float | bool]] = []
        for metadata in values:
            if not isinstance(metadata, Mapping):
                raise TypeError("Каждый элемент metadatas должен быть словарём")
            clean_metadata: dict[str, str | int | float | bool] = {}
            for key, value in metadata.items():
                if not isinstance(key, str) or not key:
                    raise TypeError("Ключи metadata должны быть непустыми строками")
                if not isinstance(value, allowed):
                    raise TypeError(
                        f"Значение metadata[{key!r}] должно быть str, int, float или bool"
                    )
                clean_metadata[key] = value
            result.append(clean_metadata)
        return result

    @staticmethod
    def _new_ids(size: int) -> list[str]:
        """Генерирует уникальные идентификаторы для записей без явного ID."""

        return [str(uuid4()) for _ in range(size)]

    # ---------------------------------------------------------------------
    # Запись и чтение
    # ---------------------------------------------------------------------

    def add(
        self,
        documents: str | Sequence[str],
        ids: str | Sequence[str] | None = None,
        metadatas: Metadata | Sequence[Metadata] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Добавляет новые документы в коллекцию.

        Если ``embeddings`` не переданы, ChromaDB вызовет настроенную
        ``OpenAIEmbeddings`` и получит векторы через OpenAI-совместимый API
        YandexGPT. Явная передача векторов полезна при пакетной обработке или
        при использовании собственной embedding-функции.

        Args:
            documents: Один текст или список текстов.
            ids: Один ID или список ID. Если не передан, ID генерируются через
                UUID4.
            metadatas: Один словарь метаданных или словарь для каждой записи.
            embeddings: Готовые векторы, по одному на документ.

        Returns:
            Список ID добавленных записей.
        """

        document_list = self._as_list(documents, "documents")
        id_list = self._new_ids(len(document_list)) if ids is None else self._as_list(ids, "ids")
        if len(id_list) != len(document_list):
            raise ValueError("Количество ids должно совпадать с количеством documents")
        metadata_list = self._normalise_metadatas(metadatas, len(document_list))
        if embeddings is not None and len(embeddings) != len(document_list):
            raise ValueError("Количество embeddings должно совпадать с количеством documents")

        kwargs: dict[str, Any] = {"ids": id_list, "documents": document_list}
        if metadata_list is not None:
            kwargs["metadatas"] = metadata_list
        if embeddings is not None:
            kwargs["embeddings"] = [list(vector) for vector in embeddings]
        self.collection.add(**kwargs)
        return id_list

    def embed_documents(
        self, texts: str | Sequence[str]
    ) -> list[list[float]]:
        """Векторизует тексты документной embedding-моделью.

        Метод полезен, когда исходный документ требуется сохранить без
        изменений, но для поиска нужен его нормализованный смысловой вариант.
        Полученные векторы можно передать в ``add()`` или ``update()``.
        """

        text_list = self._as_list(texts, "texts")
        return [list(vector) for vector in self.embedding_function(text_list)]

    def upsert(
        self,
        documents: str | Sequence[str],
        ids: str | Sequence[str],
        metadatas: Metadata | Sequence[Metadata] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Добавляет записи или заменяет записи с такими же ID.

        В отличие от ``add`` метод удобен для идемпотентной синхронизации
        профиля пользователя: повторный вызов не завершится ошибкой из-за
        уже существующего идентификатора.
        """

        document_list = self._as_list(documents, "documents")
        id_list = self._as_list(ids, "ids")
        if len(id_list) != len(document_list):
            raise ValueError("Количество ids должно совпадать с количеством documents")
        metadata_list = self._normalise_metadatas(metadatas, len(document_list))
        if embeddings is not None and len(embeddings) != len(document_list):
            raise ValueError("Количество embeddings должно совпадать с количеством documents")

        kwargs: dict[str, Any] = {"ids": id_list, "documents": document_list}
        if metadata_list is not None:
            kwargs["metadatas"] = metadata_list
        if embeddings is not None:
            kwargs["embeddings"] = [list(vector) for vector in embeddings]
        self.collection.upsert(**kwargs)
        return id_list

    def get(
        self,
        ids: str | Sequence[str] | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Читает записи по ID, метаданным или условию в тексте.

        ``where`` использует синтаксис фильтров ChromaDB, например
        ``{"user_id": "42"}`` или ``{"importance": {"$gte": 5}}``.
        Для поиска фрагмента текста можно передать
        ``where_document={"$contains": "бег"}``.

        Returns:
            Словарь ChromaDB с полями ``ids``, ``documents``, ``metadatas`` и
            другими полями из ``include``.
        """

        id_list = None if ids is None else self._as_list(ids, "ids")
        kwargs: dict[str, Any] = {
            "ids": id_list,
            "where": dict(where) if where is not None else None,
            "where_document": dict(where_document) if where_document is not None else None,
        }
        if limit is not None:
            if limit < 1:
                raise ValueError("limit должен быть больше нуля")
            kwargs["limit"] = limit
        if offset is not None:
            if offset < 0:
                raise ValueError("offset не может быть отрицательным")
            kwargs["offset"] = offset
        if include is not None:
            kwargs["include"] = list(include)
        # Удаляем None: некоторые версии Chroma строже относятся к аргументам.
        return self.collection.get(**{key: value for key, value in kwargs.items() if value is not None})

    def get_by_id(self, record_id: str) -> dict[str, Any] | None:
        """Возвращает одну запись по ID или ``None``, если такой записи нет."""

        result = self.get(record_id)
        if not result.get("ids"):
            return None
        record: dict[str, Any] = {"id": result["ids"][0]}
        for field in ("documents", "metadatas", "embeddings", "uris", "data"):
            values = result.get(field)
            if values is not None:
                record[field] = values[0]
        return record

    def list_records(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Возвращает страницу записей коллекции для просмотра или отладки."""

        return self.get(limit=limit, offset=offset)

    def count(self) -> int:
        """Возвращает текущее количество записей в коллекции."""

        return self.collection.count()

    def peek(self, limit: int = 10) -> dict[str, Any]:
        """Показывает первые записи коллекции без изменения данных."""

        if limit < 1:
            raise ValueError("limit должен быть больше нуля")
        return self.collection.peek(limit=limit)

    # ---------------------------------------------------------------------
    # Поиск
    # ---------------------------------------------------------------------

    def search(
        self,
        query_text: str | Sequence[str],
        n_results: int = 5,
        *,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Ищет наиболее близкие по смыслу документы.

        Запрос превращается в вектор с помощью OpenAI-совместимой модели
        эмбеддингов YandexGPT, после чего ChromaDB возвращает ближайшие записи.
        Можно дополнительно ограничить поиск метаданными через ``where``.

        Args:
            query_text: Один поисковый запрос или несколько запросов.
            n_results: Максимальное число результатов на каждый запрос.
            where: Фильтр по метаданным.
            where_document: Дополнительный фильтр по содержимому документов.
            include: Поля результата, например ``["documents", "distances"]``.

        Returns:
            Словарь с вложенными списками результатов ChromaDB. Поле
            ``cosine_distances`` дублирует ``distances`` в явном виде: элемент
            ``cosine_distances[i][j]`` относится к ``ids[i][j]``. Чем меньше
            значение, тем ближе результат к поисковому запросу.
        """

        query_list = self._as_list(query_text, "query_text")
        if n_results < 1:
            raise ValueError("n_results должен быть больше нуля")
        kwargs: dict[str, Any] = {
            "query_texts": query_list,
            "n_results": n_results,
        }
        if where is not None:
            kwargs["where"] = dict(where)
        if where_document is not None:
            kwargs["where_document"] = dict(where_document)
        # При обычном вызове search() нужны не только расстояния, но и сами
        # документы: answer_with_memory() использует documents для построения
        # контекста prompt. Ранее здесь из-за ``include or []`` запрашивались
        # только distances, поэтому LLM получала пустой контекст памяти.
        requested_include = (
            ["documents", "metadatas", "distances"]
            if include is None
            else list(include)
        )
        # Расстояние добавляется всегда, даже если вызывающий код явно попросил
        # только документы. Так контракт search() гарантирует расстояние для
        # каждого найденного результата.
        if "distances" not in requested_include:
            requested_include.append("distances")
        if requested_include:
            kwargs["include"] = requested_include

        result = self.collection.query(**kwargs)
        # Chroma возвращает distances параллельными списками: distances[i][j]
        # соответствует ids[i][j]. Добавляем говорящий alias, чтобы в коде
        # бота было явно видно, что это именно косинусное расстояние.
        if result.get("distances") is not None:
            result["cosine_distances"] = result["distances"]
        return result

    def search_by_embedding(
        self,
        embedding: Sequence[float] | Sequence[Sequence[float]],
        n_results: int = 5,
        *,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Ищет записи по уже готовому вектору, не вызывая API эмбеддингов."""

        vector = list(embedding)
        # Один вектор — наиболее частый случай; Chroma ожидает список векторов.
        if vector and isinstance(vector[0], (int, float)):
            vectors: list[list[float]] = [vector]  # type: ignore[list-item]
        else:
            vectors = [list(item) for item in vector]  # type: ignore[arg-type]
        if not vectors:
            raise ValueError("embedding не должен быть пустым")
        kwargs: dict[str, Any] = {
            "query_embeddings": vectors,
            "n_results": n_results,
        }
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.collection.query(**kwargs)

    def find_by_text(self, text: str, limit: int = 100) -> dict[str, Any]:
        """Ищет записи, содержащие точный фрагмент текста.

        Это не семантический поиск: ChromaDB проверяет наличие подстроки.
        Для поиска по смыслу используйте ``search``.
        """

        if not text:
            raise ValueError("text не должен быть пустым")
        return self.get(where_document={"$contains": text}, limit=limit)

    # ---------------------------------------------------------------------
    # Изменение и удаление
    # ---------------------------------------------------------------------

    def update(
        self,
        record_id: str,
        *,
        document: str | None = None,
        metadata: Metadata | None = None,
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Изменяет существующую запись по ID.

        Передайте хотя бы одно из ``document``, ``metadata`` или ``embedding``.
        Если меняется только текст, ChromaDB автоматически пересчитает вектор.
        """

        if not record_id:
            raise ValueError("record_id не должен быть пустым")
        if document is None and metadata is None and embedding is None:
            raise ValueError("Нужно передать document, metadata или embedding")

        kwargs: dict[str, Any] = {"ids": [record_id]}
        if document is not None:
            kwargs["documents"] = [document]
        if metadata is not None:
            kwargs["metadatas"] = self._normalise_metadatas(metadata, 1)
        if embedding is not None:
            kwargs["embeddings"] = [list(embedding)]
        self.collection.update(**kwargs)
        return record_id

    def delete(
        self,
        ids: str | Sequence[str] | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> None:
        """Удаляет одну или несколько записей по ID и/или условиям.

        Фильтры передаются напрямую в ChromaDB. Для удаления ровно одной
        найденной записи используйте ``delete_one`` — он сначала проверит
        количество совпадений и не удалит случайно несколько записей.
        """

        id_list = None if ids is None else self._as_list(ids, "ids")
        if id_list is None and where is None and where_document is None:
            raise ValueError("Нужно указать ids, where или where_document")
        kwargs: dict[str, Any] = {
            "ids": id_list,
            "where": dict(where) if where is not None else None,
            "where_document": dict(where_document) if where_document is not None else None,
        }
        self.collection.delete(**{key: value for key, value in kwargs.items() if value is not None})

    def delete_one(
        self,
        record_id: str | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> str:
        """Удаляет ровно одну запись и возвращает её ID.

        Можно передать ID, фильтр метаданных или фильтр текста. Если условие
        соответствует нулю или нескольким документам, метод завершится с
        ошибкой до удаления данных.
        """

        if record_id is not None:
            if where is not None or where_document is not None:
                raise ValueError("Для delete_one нельзя смешивать ID и фильтры")
            result = self.get(ids=record_id, include=[])
            matched_ids = list(result.get("ids", []))
        else:
            if where is None and where_document is None:
                raise ValueError("Для delete_one укажите record_id, where или where_document")
            result = self.get(where=where, where_document=where_document, include=[])
            matched_ids = list(result.get("ids", []))
        if len(matched_ids) != 1:
            raise ValueError(
                f"Условие должно найти ровно одну запись, найдено: {len(matched_ids)}"
            )
        self.collection.delete(ids=matched_ids)
        return matched_ids[0]

    def clear(self, batch_size: int = 500) -> int:
        """Полностью очищает коллекцию, сохраняя саму коллекцию.

        Удаление выполняется пакетами, чтобы не собирать весь список ID в один
        большой запрос. Возвращает число удалённых записей.
        """

        if batch_size < 1:
            raise ValueError("batch_size должен быть больше нуля")
        deleted = 0
        while True:
            page = self.collection.get(limit=batch_size, offset=0, include=[])
            ids = page.get("ids", [])
            if not ids:
                break
            self.collection.delete(ids=ids)
            deleted += len(ids)
            if len(ids) < batch_size:
                break
        return deleted

    def drop_collection(self, *, confirm: bool = False) -> None:
        """Удаляет коллекцию целиком.

        Операция намеренно требует ``confirm=True``. В отличие от ``clear``
        после неё исчезает объект коллекции и его настройки; при следующем
        обращении коллекция будет создана заново.
        """

        if not confirm:
            raise ValueError("Для удаления коллекции передайте confirm=True")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=self.collection_metadata,
        )

    # ---------------------------------------------------------------------
    # Удобства для памяти ассистента и YandexGPT
    # ---------------------------------------------------------------------

    def _get_llm(self) -> ChatOpenAI:
        """Создаёт и кэширует клиент ChatOpenAI при первом запросе к LLM."""

        if self.llm is None:
            self.llm = ChatOpenAI(
                model=self.chat_model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self.llm

    def remember(
        self,
        text: str,
        *,
        user_id: str | int | None = None,
        memory_type: str = "conversation",
        importance: int | float = 1,
        record_id: str | None = None,
        metadata: Metadata | None = None,
    ) -> str:
        """Сохраняет факт или фрагмент диалога в формате памяти ассистента.

        Метод добавляет служебные поля ``memory_type`` и ``created_at``. Они
        позволяют позднее фильтровать память пользователя и отделять факты,
        предпочтения, задачи и сообщения друг от друга.
        """

        if not text:
            raise ValueError("text не должен быть пустым")
        if not isinstance(importance, (int, float)):
            raise TypeError("importance должен быть числом")
        memory_metadata: dict[str, str | int | float | bool] = {
            "memory_type": memory_type,
            "importance": importance,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if user_id is not None:
            memory_metadata["user_id"] = str(user_id)
        if metadata:
            memory_metadata.update(metadata)
        return self.add(text, ids=record_id, metadatas=memory_metadata)[0]

    def search_memory(
        self,
        query: str,
        *,
        user_id: str | int | None = None,
        memory_type: str | None = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Ищет релевантные воспоминания с фильтрами пользователя и типа."""

        filters: list[dict[str, Any]] = []
        if user_id is not None:
            filters.append({"user_id": str(user_id)})
        if memory_type is not None:
            filters.append({"memory_type": memory_type})
        where: dict[str, Any] | None
        if not filters:
            where = None
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = {"$and": filters}
        return self.search(query, n_results=n_results, where=where)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Отправляет запрос в YandexGPT через OpenAI-совместимый API."""

        if not prompt:
            raise ValueError("prompt не должен быть пустым")
        messages: list[tuple[str, str]] = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self._get_llm().invoke(messages, **kwargs)
        content = response.content
        if isinstance(content, str):
            return content
        # Некоторые совместимые API могут вернуть список блоков контента.
        return "".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )

    def answer_with_memory(
        self,
        question: str,
        *,
        user_id: str | int | None = None,
        n_results: int = 5,
        system_prompt: str = (
            "Ты полезный ассистент. Отвечай на русском языке. "
            "Используй контекст памяти только если он действительно относится к вопросу."
        ),
    ) -> str:
        """Ищет память, формирует контекст и просит YandexGPT ответить.

        Это демонстрационный orchestration-метод для будущего бота. В реальном
        приложении можно отдельно контролировать форматирование контекста,
        лимит токенов и правила приватности.
        """

        results = self.search_memory(question, user_id=user_id, n_results=n_results)
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        context_items = documents[0] if documents else []
        metadata_items = metadatas[0] if metadatas else []
        context_lines: list[str] = []
        for index, document in enumerate(context_items):
            if not document:
                continue
            metadata = metadata_items[index] if index < len(metadata_items) else {}
            normalized_fact = metadata.get("normalized_fact") if metadata else None
            memory_type = str(metadata.get("memory_type", "fact")) if metadata else "fact"
            status = str(metadata.get("status", "active")) if metadata else "active"
            timestamp = ""
            if metadata:
                timestamp = str(metadata.get("updated_at") or metadata.get("created_at") or "")
            header_parts = [f"тип: {memory_type}", f"статус: {status}"]
            if timestamp:
                header_parts.append(f"сведения актуальны на: {timestamp}")
            header = f"[{' | '.join(header_parts)}]"
            if normalized_fact and normalized_fact != document:
                context_lines.append(
                    f"- {header}\n  Исходная фраза: {document}\n"
                    f"  Нормализованная запись: {normalized_fact}"
                )
            else:
                context_lines.append(f"- {header}\n  {document}")
        context = "\n\n".join(context_lines)
        if not context:
            context = "Память по этому вопросу не найдена."
        prompt = f"Контекст памяти:\n{context}\n\nВопрос пользователя:\n{question}"
        return self.generate(prompt, system_prompt=system_prompt)


# Псевдоним с более коротким названием оставлен для удобного импорта в боте.
VectorStoreManager = ChromaManager


__all__ = ["ChromaManager", "VectorStoreManager", "Metadata"]
