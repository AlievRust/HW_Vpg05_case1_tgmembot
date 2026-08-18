# Небольшой production-like образ для учебного синхронного Telegram-бота.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY chroma_manager.py bot.py ./

# Каталог data монтируется из docker-compose и содержит только локальную
# персистентную базу ChromaDB, а не секреты.
RUN mkdir -p /app/data

CMD ["python", "bot.py"]
