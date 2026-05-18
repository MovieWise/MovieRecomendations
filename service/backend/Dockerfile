FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/service/backend

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY service/backend/requirements312.txt service/backend/requirements312.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r service/backend/requirements312.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "recommendation_service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
