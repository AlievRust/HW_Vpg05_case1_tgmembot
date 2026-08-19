# Telegram-бот-помощник с векторной памятью

Учебный проект Telegram-бота на `pyTelegramBotAPI`. Бот общается с пользователем, ведёт универсальную долговременную память в ChromaDB и использует YandexGPT через OpenAI-совместимый API.

## Возможности

- синхронный Telegram polling без асинхронной логики;
- долговременная локальная память пользователя в ChromaDB: факты, актуальные состояния, инструкции, задачи и заметки;
- семантический поиск по сохранённым сообщениям;
- определение отношения новой фразы к памяти через YandexGPT:
  - `new` — новая информация;
  - `same` — уже известная информация, запись пропускается;
  - `contradiction` — противоречие, старая запись обновляется;
- предварительный анализ сообщения: вопросы и команды не попадают в
  долговременную память;
- обработка смешанных сообщений, где в одной фразе есть и факт, и вопрос;
- команда подтверждённой полной очистки памяти текущего пользователя;
- контекст ответа из нескольких похожих записей с типом, статусом и датой актуальности;
- активные инструкции пользователя автоматически добавляются в системные правила ответа;
- косинусное расстояние в результатах поиска;
- изоляция памяти пользователей по `user_id`;
- логирование действий `created`, `skipped`, `updated`;
- запуск одной командой через Docker Compose.

## Структура проекта

```text
.
├── bot.py                 # Telegram-бот и логика памяти
├── chroma_manager.py      # Универсальный менеджер ChromaDB и YandexGPT
├── requirements.txt       # Зависимости Python
├── Dockerfile             # Образ приложения
├── docker-compose.yml     # Запуск приложения и volume для памяти
├── .env.example           # Пример конфигурации без секретов
└── data/                  # Локальная база ChromaDB, создаётся при запуске
```

## Как работает память

Для каждого текстового сообщения пользователя выполняется следующий сценарий:

```text
сообщение пользователя
        │
        ▼
локальный prefilter и LLM-анализ claims
        │
        ├── вопрос/команда/приветствие ────► skipped
        │
        └── найдены одна или несколько записей памяти
                    │
                    ▼
          семантический поиск записи того же типа
                    │
          ├── нет близкого результата ─────► created
          │
          └── найден кандидат
                    │
                    ▼
             YandexGPT-классификатор
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        same       new   contradiction
          │         │          │
       skipped   created     updated
```

В документ ChromaDB записывается только исходный текст сообщения пользователя. Ответ бота, классификационный prompt и шаблонные фразы в документ памяти не сохраняются. Одно сообщение может дать несколько атомарных записей: например, факт о переезде и отдельную задачу.

Если сообщение смешанное, например `Я живу в Екатеринбурге, а где живёшь ты?`, LLM извлекает утверждение и пропускает вопрос. Исходная фраза сохраняется как `document`, а нормализованный факт и список claims — в metadata.

LLM относит каждую запись к одному из типов:

- `fact` — устойчивый факт или справочная информация;
- `state` — текущее изменяемое состояние пользователя или другой сущности: место проживания, цена, статус;
- `instruction` — правило поведения ассистента в этом диалоге;
- `task` — задача, намерение или напоминание;
- `note` — явно сохранённая свободная заметка.

Для каждой записи дополнительно используются metadata:

- `user_id` — идентификатор Telegram-пользователя;
- `memory_type` — тип записи из списка выше;
- `subject`, `predicate`, `value` — нормализованная структура записи;
- `created_at` — время создания;
- `updated_at` — время последнего обновления;
- `valid_from` и `valid_to` — период актуальности для `state`;
- `status` — текущий статус записи (`active`).

При противоречии для `state` сохраняется прежняя логика проекта: текущая запись обновляется. Поэтому история предыдущих значений пока не ведётся. Например, после переезда запись о Екатеринбурге заменяется записью о Берлине, но дата `updated_at` позволяет LLM учитывать свежесть результата.

Инструкции не смешиваются с профилем пользователя. Фраза «Веди себя как Гомер Симпсон» сохраняется как `instruction` и при каждом следующем сообщении подаётся модели вместе с основным системным prompt. Нейтральная заметка вроде «Билет Екатеринбург—Берлин стоит 30 000 рублей» хранится как отдельный факт или актуальное состояние билета и не требует специальной ветки кода.

## Требования

- Python 3.12+ для локального запуска;
- Docker и Docker Compose для контейнерного запуска;
- Telegram Bot Token;
- API-ключ Yandex AI Studio;
- каталог Yandex Cloud с доступом к моделям YandexGPT и embeddings.

Yandex AI Studio используется через OpenAI-совместимый endpoint:

```text
https://ai.api.cloud.yandex.net/v1
```

Документация: [OpenAI-compatible API Yandex AI Studio](https://aistudio.yandex.ru/docs/en/ai-studio/api/) и [модели векторизации текста](https://aistudio.yandex.ru/docs/en/ai-studio/concepts/embeddings.html).

## Настройка `.env`

Скопируйте пример конфигурации:

```bash
copy .env.example .env
```

Для Linux/macOS:

```bash
cp .env.example .env
```

Заполните секреты:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

YANDEX_API_KEY=your_yandex_api_key
YANDEX_FOLDER_ID=your_folder_id
YANDEX_OPENAI_BASE_URL=https://ai.api.cloud.yandex.net/v1
YANDEXGPT_MODEL=gpt://your_folder_id/yandexgpt/latest
YANDEX_EMBEDDING_MODEL=emb://your_folder_id/text-search-doc/latest
YANDEX_QUERY_EMBEDDING_MODEL=emb://your_folder_id/text-search-query/latest
```

### Основные переменные

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `TELEGRAM_BOT_TOKEN` | — | Обязательный токен Telegram-бота |
| `YANDEX_API_KEY` | — | API-ключ Yandex AI Studio |
| `YANDEX_FOLDER_ID` | `<folder_id>` | ID каталога Yandex Cloud |
| `YANDEX_OPENAI_BASE_URL` | `https://ai.api.cloud.yandex.net/v1` | OpenAI-совместимый endpoint |
| `YANDEXGPT_MODEL` | `gpt://<folder_id>/yandexgpt/latest` | Модель генерации и классификации |
| `YANDEX_EMBEDDING_MODEL` | `emb://<folder_id>/text-search-doc/latest` | Модель embeddings |
| `YANDEX_QUERY_EMBEDDING_MODEL` | `emb://<folder_id>/text-search-query/latest` | Модель embeddings для поисковых запросов |
| `CHROMA_COLLECTION` | `assistant_memory` | Имя коллекции ChromaDB |
| `CHROMA_PERSIST_DIRECTORY` | `data/chroma` | Каталог локального хранилища |
| `MEMORY_CONTEXT_SIZE` | `5` | Количество записей в контексте ответа |
| `MEMORY_SEARCH_CANDIDATES` | `5` | Количество кандидатов для проверки памяти |
| `MEMORY_MAX_DISTANCE` | `0.8` | Максимальное расстояние для передачи кандидата в LLM-классификатор |
| `MEMORY_MIN_CONFIDENCE` | `0.7` | Минимальная уверенность LLM для сохранения факта |
| `BOT_SYSTEM_PROMPT` | встроенный prompt | Системная инструкция ассистента |
| `LOG_LEVEL` | `INFO` | Уровень логирования |

Проект также понимает старые имена из существующей конфигурации: `YANDEX_CLOUD_MODEL`, `YANDEX_OPENAI_BASE_URL`, `CHROMA_DIR` и `TOP_K`.

## Запуск через Docker Compose

После заполнения `.env` выполните:

```bash
docker compose up --build
```

Фоновый запуск:

```bash
docker compose up --build -d
```

Просмотр логов:

```bash
docker compose logs -f telegram-memory-bot
```

Остановка:

```bash
docker compose down
```

Данные ChromaDB сохраняются в локальном каталоге `data/`, подключённом в контейнер как `/app/data`. Удаление контейнера не удаляет эту память.

## Локальный запуск

Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python bot.py
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python bot.py
```

## Логи памяти

При обработке сообщения в консоли появляются записи вида:

```text
action: created user_id=123 record_id=... distance=None
action: skipped user_id=123 record_id=... distance=0.02 reason=...
action: updated user_id=123 record_id=... distance=0.18 reason=...
```

Если бот создаёт слишком много повторов, увеличьте `MEMORY_MAX_DISTANCE`, чтобы больше кандидатов проверялось через LLM. Если в классификатор попадает слишком много явно нерелевантных фактов, уменьшите это значение.

## Управление памятью пользователя

Для удаления всей долговременной памяти текущего пользователя:

```text
/forget_me
```

После этого бот попросит подтверждение:

```text
/forget_me_confirm
```

Также поддерживается alias `/clear_memory`. Удаление выполняется только для Telegram-пользователя, отправившего команду.

## Основные классы и методы

### `ChromaManager`

Находится в `chroma_manager.py` и предоставляет:

- `add()` и `upsert()` — запись документов;
- `get()` и `get_by_id()` — чтение;
- `search()` — семантический поиск с `distances` и `cosine_distances`;
- `search_memory()` — поиск с фильтрацией по пользователю;
- `update()` — обновление записи;
- `delete()` и `delete_one()` — удаление;
- `clear()` — очистка коллекции;
- `remember()` — удобное сохранение факта;
- `generate()` — вызов YandexGPT;
- `answer_with_memory()` — ответ с найденным контекстом.

### `MemoryService`

Находится в `bot.py` и связывает Telegram-сообщение с менеджером памяти:

```python
service.remember_message("123456", "Я живу в Екатеринбурге")
answer = service.answer("123456", "Где я живу?")
```

## Безопасность и ограничения учебного проекта

- Не передавайте `.env` в Git и не встраивайте секреты в Dockerfile.
- Память хранится локально в ChromaDB без отдельного шифрования.
- Для минимальной нагрузки обработка выполняется последовательно.
- На каждое сообщение могут приходиться запросы к embeddings и YandexGPT.
- Решение о противоречии зависит от качества ответа LLM.
- Для production-проекта понадобятся rate limit, обработка Telegram retry, шифрование данных, резервное копирование и более строгая политика хранения персональных данных.

## Проверка проекта

Проверка синтаксиса:

```bash
python -m py_compile chroma_manager.py bot.py
```

Проверка Compose-конфигурации:

```bash
docker compose config --quiet
```

## Диагностика embeddings

Если в логах появляется:

```text
POST https://ai.api.cloud.yandex.net/v1/embeddings 400 Bad Request
Failed to parse request JSON
```

убедитесь, что контейнер пересобран после обновления `chroma_manager.py`:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

Текущая реализация отправляет исходный текст напрямую в embeddings API как строковый `input`, а не как массив token IDs. В `.env` желательно явно указать обе модели:

```env
YANDEX_EMBEDDING_MODEL=emb://your_folder_id/text-search-doc/latest
YANDEX_QUERY_EMBEDDING_MODEL=emb://your_folder_id/text-search-query/latest
```
