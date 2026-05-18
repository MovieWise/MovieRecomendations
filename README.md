# MovieRecs - гибридная рекомендательная система фильмов

MovieRecs - проект рекомендательной системы фильмов с ML-ядром, FastAPI backend, Telegram Mini App frontend и legacy Streamlit-интерфейсом. Основной пользовательский сценарий: пользователь открывает Telegram Mini App, отмечает просмотренные фильмы лайком или дизлайком, а приложение строит персональную ленту рекомендаций через гибридную модель `EASE + LightGBM`.

---

## Состав команды:
- Иванов Михаил
- Власов Никита

## Куратор:
- Ляпин Данила

---

## 🎯 Цели проекта
1. Построить и сравнить **базовые и продвинутые модели рекомендаций**.  
2. Реализовать **гибридную систему**, объединяющую данные о пользователях и контенте фильмов.  
3. Оценить качество разных подходов с использованием современных **метрик** (*Precision@K, Recall@K, NDCG*).  
4. Разработать **прототип веб-приложения** для демонстрации рекомендаций.  

---

## 📂 Данные
В проекте используются открытые датасеты:  
- **[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)** — оценки пользователей, теги, ID фильмов.  
- **[TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)** — метаданные о фильмах (жанры, описания, актёры, бюджеты, сборы).  
- **[IMDb Datasets](https://www.imdb.com/interfaces/)** — дополнительная информация об актёрах и режиссёрах.

- Очищенные и подготовленные данные представлены по **[ссылке](https://disk.360.yandex.ru/d/tgOlYgExcd9S0A)**

---

## Архитектура

- `src/movie_recs` - reusable ML-пакет: подготовка данных, модели, метрики, обучение и inference.
- `service/backend/recommendation_service` - FastAPI backend с API `/api/v1`.
- `service/frontend/telegram-mini-app` - React, TypeScript и Vite frontend для Telegram Mini App.
- `service/frontend/app.py` - legacy Streamlit UI, оставлен для совместимости.
- `service/backend/recommendation_service/data` - локальная папка для CSV, parquet, SQLite DB и model artifacts. Эти файлы не коммитятся.

Backend разделен по слоям:

- `api/routers` - HTTP endpoints.
- `schemas` - Pydantic request и response модели.
- `services` - бизнес-логика каталога, реакций, OMDb и рекомендаций.
- `repositories` - SQLAlchemy доступ к пользователям, реакциям и OMDb cache.
- `ml` - загрузка артефактов и inference `EASE + LightGBM`.
- `clients` - внешние API clients.
- `core` - config, logging и security.

## Основные возможности

- Авторизация Telegram Mini App через проверку Telegram WebApp init data.
- JWT для защищенных backend endpoints.
- Каталог популярных фильмов для холодного старта.
- Поиск фильмов по названию.
- Сохранение лайков и дизлайков с возможностью перезаписи и удаления оценки.
- Автоматическая персональная лента рекомендаций после появления пользовательских оценок.
- OMDb-обогащение карточек: poster, title, year, genre, plot, rating, runtime, director, actors, imdbID.
- Кэширование OMDb-ответов в базе.
- Legacy Streamlit endpoints сохранены.

## Backend API

Основные endpoints:

- `POST /api/v1/auth/telegram` - проверка Telegram init data и выдача JWT.
- `GET /api/v1/movies/feed` - фильмы для оценки, исключая уже оцененные.
- `GET /api/v1/movies/search?q=...` - поиск по каталогу.
- `GET /api/v1/movies/{movie_id}` - подробная карточка фильма.
- `POST /api/v1/reactions` - создать или обновить лайк/дизлайк.
- `DELETE /api/v1/reactions/{movie_id}` - удалить оценку.
- `GET /api/v1/profile/ratings` - список пользовательских оценок.
- `POST /api/v1/recommendations/generate` - сгенерировать рекомендации.
- `GET /api/v1/health` - проверка API, конфигурации, БД и ML-артефактов.

Legacy endpoints `/forward`, `/predict_raw`, `/history` и `/stats` сохранены для Streamlit.

## Данные и артефакты

Для каталога и холодного старта нужен файл:

- `service/backend/recommendation_service/data/links.csv`

Ожидаемые колонки для каталога:

- `movieId`
- `imdbId`
- `title`

Для рекомендаций через `EASE + LightGBM` нужны:

- `service/backend/recommendation_service/data/ease_weights_f16.npy`
- `service/backend/recommendation_service/data/ease_item_encoder.joblib`
- `service/backend/recommendation_service/data/ease_interaction_matrix.npz`
- `service/backend/recommendation_service/data/LightGBMHybridRanker.pkl`
- `service/backend/recommendation_service/data/content.parquet`

Дополнительно поддерживаются legacy-имена и артефакты:

- `service/backend/recommendation_service/data/item_encoder.joblib`
- `service/backend/recommendation_service/data/ease_user_encoder.joblib`
- `service/backend/recommendation_service/data/puresvd_data.pkl`
- `service/backend/recommendation_service/data/mostpop_data.pkl`

Если обязательные ML-артефакты отсутствуют, healthcheck покажет, что модель недоступна, а endpoint рекомендаций вернет понятную ошибку `model_unavailable`.

## Переменные окружения backend

Скопируйте пример:

```bash
cp service/backend/recommendation_service/.env.example service/backend/recommendation_service/.env
```

Заполните значения:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
JWT_SECRET=change-me-long-random-secret
JWT_TTL_SECONDS=604800
TELEGRAM_AUTH_MAX_AGE_SECONDS=86400

DATABASE_URL=sqlite:///./service/backend/recommendation_service/data/database.db

OMDB_API_KEY=your_omdb_api_key
OMDB_BASE_URL=https://www.omdbapi.com/
OMDB_CACHE_TTL_SECONDS=604800

MOVIES_PATH=service/backend/recommendation_service/data/links.csv
LINKS_PATH=service/backend/recommendation_service/data/links.csv
CONTENT_FEATURES_PATH=service/backend/recommendation_service/data/content.parquet

EASE_WEIGHTS_PATH=service/backend/recommendation_service/data/ease_weights_f16.npy
EASE_ITEM_ENCODER_PATH=service/backend/recommendation_service/data/ease_item_encoder.joblib
EASE_USER_ENCODER_PATH=service/backend/recommendation_service/data/ease_user_encoder.joblib
EASE_INTERACTIONS_PATH=service/backend/recommendation_service/data/ease_interaction_matrix.npz
LGBM_RANKER_PATH=service/backend/recommendation_service/data/LightGBMHybridRanker.pkl
```

Для Telegram Mini App ключ OMDb должен быть только в backend environment. Frontend не должен получать `OMDB_API_KEY`.

## Переменные окружения frontend

Пример находится в `service/frontend/telegram-mini-app/.env.example`.

```env
VITE_API_BASE_URL=/api/v1
VITE_BACKEND_PROXY_TARGET=http://localhost:8000
VITE_TELEGRAM_INIT_DATA=
```

`VITE_TELEGRAM_INIT_DATA` нужен только для локального smoke-теста в браузере вне Telegram.

## Запуск без Docker

Установите Python-зависимости:

```bash
pip install -r requirements.txt
pip install -r service/backend/requirements312.txt
```

Подготовьте backend environment и запустите FastAPI:

```bash
set -a
source service/backend/recommendation_service/.env
set +a
export PYTHONPATH="$PWD/src:$PWD/service/backend"
uvicorn recommendation_service.api.main:app --reload --host 0.0.0.0 --port 8000
```

Проверьте backend:

```bash
curl http://localhost:8000/api/v1/health
```

Запустите Telegram Mini App frontend:

```bash
cd service/frontend/telegram-mini-app
npm install
npm run dev
```

Откройте `http://localhost:5173`.

## Локальный smoke-тест без Telegram

Нужно сгенерировать `VITE_TELEGRAM_INIT_DATA` тем же `TELEGRAM_BOT_TOKEN`, который указан в backend `.env`.

```bash
python - <<'PY'
import hashlib
import hmac
import json
import os
import time
import urllib.parse

bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
user = {
    "id": 100001,
    "first_name": "Local",
    "last_name": "Tester",
    "username": "local_tester",
    "language_code": "ru",
}
params = {
    "query_id": "local-smoke",
    "user": json.dumps(user, separators=(",", ":")),
    "auth_date": str(int(time.time())),
}
data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(params.items()))
secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
params["hash"] = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
print(urllib.parse.urlencode(params))
PY
```

Затем перезапустите Vite с этим значением:

```bash
cd service/frontend/telegram-mini-app
VITE_API_BASE_URL=/api/v1 VITE_TELEGRAM_INIT_DATA='generated_init_data_here' npm run dev
```

## Запуск через Docker Compose

Перед запуском создайте backend `.env` и положите data/model artifacts в `service/backend/recommendation_service/data`.

```bash
cp service/backend/recommendation_service/.env.example service/backend/recommendation_service/.env
docker compose up --build
```

После запуска:

- backend: `http://localhost:8000/api/v1/health`
- frontend: `http://localhost:5173`

Compose запускает frontend как Vite dev server. Vite проксирует `/api/*` в контейнер `backend:8000`, поэтому frontend использует `VITE_API_BASE_URL=/api/v1`.

## Запуск в Telegram через один ngrok tunnel

Запустите backend и frontend локально или через Docker Compose.

Откройте один tunnel только на Vite:

```bash
ngrok http 5173
```

В BotFather укажите HTTPS URL из ngrok как Web App или Menu Button URL. Backend наружу отдельно открывать не нужно: Telegram открывает frontend, а Vite dev server проксирует `/api/*` во внутренний backend.

Для Telegram-сценария:

- `TELEGRAM_BOT_TOKEN` в backend `.env` должен совпадать с токеном вашего бота.
- `VITE_TELEGRAM_INIT_DATA` не нужен.
- `OMDB_API_KEY` остается только в backend `.env`.

## Legacy Streamlit

Streamlit-прототип расположен в `service/frontend/app.py`.

```bash
streamlit run service/frontend/app.py
```

Локальный файл `service/frontend/.streamlit/secrets.toml` не должен попадать в GitHub. Используйте `service/frontend/.streamlit/secrets.example.toml` как шаблон.

## Тесты и проверки

Backend и ML tests:

```bash
pytest -q
```

Frontend build:

```bash
cd service/frontend/telegram-mini-app
npm run build
```

Docker Compose config:

```bash
docker compose config
```

Полезная ручная проверка:

- открыть Telegram Mini App или локальный smoke-тест;
- авторизоваться через Telegram init data;
- найти фильм;
- поставить лайк или дизлайк;
- открыть список оценок;
- удалить оценку;
- проверить, что после оценок главная лента показывает персональные рекомендации;
- открыть подробную карточку рекомендованного фильма.

## GitHub-памятка

Не коммитьте:

- `.env` и `.env.*`, кроме `.env.example`;
- `service/frontend/.streamlit/secrets.toml`;
- SQLite DB;
- CSV, parquet и model artifacts;
- `node_modules`;
- frontend `dist`;
- Python caches и virtualenv.

Коммитьте:

- исходный код;
- тесты;
- Dockerfile и `docker-compose.yml`;
- `.env.example`;
- `secrets.example.toml`;
- документацию.
