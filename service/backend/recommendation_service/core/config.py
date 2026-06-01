from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVICE_ROOT.parents[2]


def _path_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "MovieRecs Telegram API"
    api_prefix: str = "/api/v1"
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{SERVICE_ROOT / 'data' / 'database.db'}",
    )
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-only-change-me")
    jwt_ttl_seconds: int = int(os.getenv("JWT_TTL_SECONDS", "604800"))
    telegram_auth_max_age_seconds: int = int(os.getenv("TELEGRAM_AUTH_MAX_AGE_SECONDS", "86400"))
    omdb_api_key: str = os.getenv("OMDB_API_KEY", "")
    omdb_base_url: str = os.getenv("OMDB_BASE_URL", "https://www.omdbapi.com/")
    omdb_cache_ttl_seconds: int = int(os.getenv("OMDB_CACHE_TTL_SECONDS", "604800"))
    movies_path: str = _path_env("MOVIES_PATH", "service/backend/recommendation_service/data/links.csv")
    links_path: str = _path_env("LINKS_PATH", "service/backend/recommendation_service/data/links.csv")
    content_features_path: str = _path_env(
        "CONTENT_FEATURES_PATH",
        "service/backend/recommendation_service/data/content.parquet",
    )
    ease_weights_path: str = _path_env(
        "EASE_WEIGHTS_PATH",
        "service/backend/recommendation_service/data/ease_weights_f16.npy",
    )
    ease_item_encoder_path: str = _path_env(
        "EASE_ITEM_ENCODER_PATH",
        "service/backend/recommendation_service/data/ease_item_encoder.joblib",
    )
    ease_user_encoder_path: str = _path_env(
        "EASE_USER_ENCODER_PATH",
        "service/backend/recommendation_service/data/ease_user_encoder.joblib",
    )
    ease_interactions_path: str = _path_env(
        "EASE_INTERACTIONS_PATH",
        "service/backend/recommendation_service/data/ease_interaction_matrix.npz",
    )
    lgbm_ranker_path: str = _path_env(
        "LGBM_RANKER_PATH",
        "service/backend/recommendation_service/data/LightGBMHybridRanker.pkl",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
