# MLflow PRD workflow

Этот каталог содержит чистые блокноты для финального наблюдаемого эксперимента `EASE + LightGBM Ranker`.

## Локальная инфраструктура

```bash
docker compose -f docker-compose.mlflow.yml up -d
```

MLflow UI: `http://localhost:5000`

MinIO console: `http://localhost:9001`

Локальные переменные окружения:

```bash
set -a
source configs/mlflow.env.example
set +a
export PYTHONPATH="$PWD/src"
```

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

Для MLflow-нужен Docker Desktop или совместимый Docker Engine. После установки Docker:

```bash
python -m movie_recs.cli.train \
  --config configs/experiments/hybrid/lgb_ranker.yaml \
  --mlflow
```

Для быстрого smoke-теста:

```bash
python -m movie_recs.cli.train \
  --config configs/experiments/hybrid/lgb_ranker.yaml \
  --mlflow \
  --limit-users 200
```

`data/processed/hybrid_train.parquet` используется для train/validation, `data/processed/hybrid_test.parquet` используется для финальных test-метрик.

## Блокноты

- `01_train_prd_ease_lgb_ranker.ipynb` - обучение и логирование финального PRD run.
- `02_error_analysis_prd.ipynb` - загрузка error и robustness artifacts из MLflow.
- `03_load_prd_model_predict.ipynb` - загрузка PRD модели из MLflow и тестовый predict.
