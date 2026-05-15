# 🎬 MovieRecs — Гибридная рекомендательная система фильмов

## 📖 Описание проекта
Этот проект посвящён разработке рекомендательной системы для фильмов на основе пользовательских предпочтений и метаданных о фильмах.  
В отличие от классических решений, основанных только на рейтингах, в данном проекте реализован **гибридный подход**, объединяющий:  
- **коллаборативную фильтрацию** (рекомендации на основе истории просмотров и оценок пользователей),  
- **контентные признаки** фильмов (жанры, актёры, описания, год выпуска, бюджет и т.д.).  

Такой метод позволяет повысить точность рекомендаций и решать проблему **холодного старта**.  

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

## 🛠️ Методы и модели

### 🔹 Базовые модели
- **Популярность** (Top-N фильмов)  
- **User-based Collaborative Filtering**  
- **Item-based Collaborative Filtering**  

### 🔹 Продвинутые модели
- **Matrix Factorization (ALS, SVD)**  
- **LightFM** (гибридная модель)  
- **Neural Collaborative Filtering (NCF)**  

### 🔹 Гибридная система
- Объединение факторов пользователей и признаков фильмов  
- Векторизация описаний фильмов (**TF-IDF / Word2Vec / BERT**)  
- Совмещение с матричной факторизацией для финального ранжирования  

---

## 📊 Метрики оценки
Для оценки качества моделей используются метрики ранжирования:  
- **Precision@K**  
- **Recall@K**  
- **Mean Average Precision (MAP)**  
- **Normalized Discounted Cumulative Gain (NDCG)**  

---

## 💻 Визуализация и интерфейс
Для демонстрации работы системы будет создано простое веб-приложение на **Streamlit**, где пользователь сможет:  
- выбрать любимые фильмы,  
- получить персонализированные рекомендации.  

---

## 🚀 План проекта
1. 📂 Сбор и предобработка данных  
2. 📊 Разведочный анализ данных (EDA)  
3. 🔧 Реализация базовых моделей  
4. ⚡ Реализация продвинутых методов (ALS, LightFM, NCF)  
5. 🔗 Построение гибридной системы  
6. 📈 Оценка качества и сравнение подходов  
7. 🎨 Визуализация и веб-приложение  
8. 📝 Подготовка итогового отчёта  

---

## 📌 Итог
Проект продемонстрирует, как можно строить **гибридные рекомендательные системы для фильмов**, объединяя пользовательские рейтинги и контентные данные. Такой подход позволяет достигать более высокой точности и делать рекомендации релевантными даже для **новых фильмов** или пользователей с небольшой историей.  

---

## Telegram Mini App

В репозитории добавлен production-oriented Telegram Mini App, который живёт рядом с legacy Streamlit-интерфейсом и переиспользует ML-код из `src/movie_recs`.

### Backend

Новый FastAPI backend расположен в `service/backend/recommendation_service` и разделён по слоям:

- `api/routers` — HTTP endpoints `/api/v1`;
- `schemas` — Pydantic-схемы;
- `services` — бизнес-логика профиля, OMDB, каталога и рекомендаций;
- `repositories` — SQLAlchemy-доступ к пользователям, реакциям, OMDB cache и истории;
- `ml` — inference `EASE + LightGBM`;
- `clients` — внешний OMDB client;
- `core` — config и security.

Основные endpoints:

- `POST /api/v1/auth/telegram`
- `GET /api/v1/movies/feed`
- `GET /api/v1/movies/search?q=...`
- `POST /api/v1/reactions`
- `DELETE /api/v1/reactions/{movie_id}`
- `GET /api/v1/profile/ratings`
- `POST /api/v1/recommendations/generate`
- `GET /api/v1/movies/{movie_id}`
- `GET /api/v1/health`

Legacy endpoints `/forward`, `/predict_raw` и `/history` сохранены для текущего Streamlit-приложения.

### Environment variables

Пример находится в `service/backend/recommendation_service/.env.example`.

Обязательные для Telegram Mini App:

- `TELEGRAM_BOT_TOKEN`
- `JWT_SECRET`
- `DATABASE_URL`
- `OMDB_API_KEY`
- `MOVIES_PATH`
- `LINKS_PATH`
- `CONTENT_FEATURES_PATH`
- `EASE_WEIGHTS_PATH`
- `EASE_ITEM_ENCODER_PATH`
- `EASE_USER_ENCODER_PATH`
- `EASE_INTERACTIONS_PATH`
- `LGBM_RANKER_PATH`

Если артефакты `EASE + LightGBM` отсутствуют, `/api/v1/health` покажет `model_available=false`, а генерация вернёт `model_unavailable` без скрытого fallback на другую модель.

### Артефакты для рекомендаций

Для отображения каталога достаточно `service/backend/recommendation_service/data/links.csv` с колонками `movieId`, `imdbId`, `title`.

Для генерации рекомендаций через `EASE + LightGBM` нужны:

- `EASE_WEIGHTS_PATH` — `ease_weights_f16.npy`;
- `EASE_ITEM_ENCODER_PATH` — `ease_item_encoder.joblib` или legacy `item_encoder.joblib`;
- `EASE_INTERACTIONS_PATH` — `ease_interaction_matrix.npz`;
- `LGBM_RANKER_PATH` — pickle-файл `LightGBMHybridRanker`;
- `CONTENT_FEATURES_PATH` — `content.parquet` или `.csv` с признаками для reranking.

`EASE_USER_ENCODER_PATH` нужен для совместимости и health/config, но текущий Telegram flow строит профиль нового пользователя по лайкам и использует item encoder.

### Запуск backend

```bash
pip install -r requirements.txt
pip install -r service/backend/requirements312.txt
export PYTHONPATH="$PWD/src:$PWD/service/backend"
uvicorn recommendation_service.api.main:app --reload --app-dir service/backend
```

### Telegram frontend

React + TypeScript + Vite приложение находится в `service/frontend/telegram-mini-app`.

```bash
cd service/frontend/telegram-mini-app
npm install
VITE_API_BASE_URL=/api/v1 npm run dev
```

Для локального smoke-теста вне Telegram можно передать `VITE_TELEGRAM_INIT_DATA`, сгенерированный под ваш `TELEGRAM_BOT_TOKEN`.

Для теста через Telegram с одним бесплатным ngrok-туннелем поднимайте туннель только на Vite:

```bash
ngrok http 5173
```

Vite проксирует `/api/*` в локальный backend на `http://localhost:8000`, поэтому второй tunnel для backend не нужен.

### Legacy Streamlit

Streamlit-прототип сохранён в `service/frontend/app.py`.

```bash
streamlit run service/frontend/app.py
```

Локальный файл `service/frontend/.streamlit/secrets.toml` не должен попадать в GitHub. Используйте `service/frontend/.streamlit/secrets.example.toml` как шаблон.

### Проверка

```bash
pytest -q
```

На текущем наборе тестов ожидается зелёный прогон; deep-тесты могут быть skipped, если опциональные DL-зависимости не установлены.
