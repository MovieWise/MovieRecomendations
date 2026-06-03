# MLflow PRD workflow

Этот каталог содержит чистые блокноты для финального наблюдаемого эксперимента `EASE + LightGBM Ranker`.

## Запуск локальной инфраструктуры

Из корня проекта:

```bash
docker compose -f docker-compose.mlflow.yml up -d
```
После запуска доступны:

- MLflow UI: `http://localhost:5000`
- MinIO console: `http://localhost:9001`

## Настройка переменных окружения

Из корня проекта:

```bash
export PYTHONPATH="$PWD/src" 

export MLFLOW_TRACKING_URI=http://localhost:5000 
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

export AWS_ACCESS_KEY_ID=minio 
export AWS_SECRET_ACCESS_KEY=minio123 
export AWS_DEFAULT_REGION=us-east-1
```

Важно:

- localhost:9000 используется как S3 API endpoint для MLflow;
- localhost:9001 — только веб-консоль MinIO.

## Запуск обучения

Для текущего PRD-конфига нужны:

- `data/processed/hybrid_train.parquet` - train/validation candidates;
- `data/processed/hybrid_test.parquet` - final test candidates;
- `data/raw/ratings.csv` - история пользователя для leakage-free счетчиков жанра, региона и кластера.

`ratings.csv` должен содержать колонки `userId`, `movieId`, `rating`, `timestamp`. Если нужно временно запустить старый feature set без пользовательских счетчиков, поставьте в `configs/experiments/hybrid/lgb_ranker.yaml`:

```yaml
training_params:
  user_counter_features:
    enabled: false
```

Если Docker не установлен, сначала можно проверить сам training без MLflow. В этом режиме артефакты сохранятся локально в `artifacts/`, но не попадут в MLflow/MinIO:

```bash
export PYTHONPATH="$PWD/src"
python -m movie_recs.cli.train \
  --config configs/experiments/hybrid/lgb_ranker.yaml \
  --limit-users 200
```

Для MLflow необходим Docker Desktop или совместимый Docker Engine. После установки Docker:

```bash
python -m movie_recs.cli.train \
  --config configs/experiments/hybrid/lgb_ranker.yaml \
  --mlflow
```

После запуска:

- run появится в MLflow UI;
- метрики будут залогированы в MLflow;
- модель будет зарегистрирована в Model Registry;
- artifacts будут сохранены в MinIO bucket `mlflow-artifacts`.

Для быстрого smoke-теста:

```bash
python -m movie_recs.cli.train \
  --config configs/experiments/hybrid/lgb_ranker.yaml \
  --mlflow \
  --limit-users 200
```

## Проверка результатов

После успешного запуска необходимо убедиться, что:

- в MLflow UI появился новый run;
- в MinIO bucket `mlflow-artifacts` сохранены артефакты;
- модель зарегистрирована в MLflow Model Registry;
- notebook `03_load_prd_model_predict.ipynb` успешно загружает модель и выполняет предсказание.

## Используемые датасеты

- `data/processed/hybrid_train.parquet` используется для train/validation, 
- `data/processed/hybrid_test.parquet` используется для финальных test-метрик.

## Блокноты

- `01_train_prd_ease_lgb_ranker.ipynb` - обучение и логирование финального PRD run.
- `02_error_analysis_prd.ipynb` - загрузка error и robustness artifacts из MLflow.
- `03_load_prd_model_predict.ipynb` - загрузка PRD модели из MLflow и тестовый predict.
