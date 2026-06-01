"""Manual smoke check for legacy model artifacts.

Run directly from the service directory when `MODEL_DATA_DIR` contains the
legacy pickle/joblib artifacts. Pytest should not execute this script.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from recommendation_service.core.model_manager import ModelManager


__test__ = False


def main() -> None:
    model = ModelManager().load_mostpop()
    print("Статистика:", model.get_user_stats())
    test_user_id = 28
    try:
        recommendations = model.predict_mostpop(test_user_id, top_n=5)
        print(f"Рекомендации для пользователя {test_user_id}: {recommendations}")
    except ValueError as exc:
        print(f"Ошибка: {exc}")
        print("Попробуй другой user_id из train_df['userId']")


if __name__ == "__main__":
    main()
