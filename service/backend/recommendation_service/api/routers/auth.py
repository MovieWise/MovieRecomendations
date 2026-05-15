from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from recommendation_service.api.deps import get_database, settings
from recommendation_service.core.config import Settings
from recommendation_service.repositories.users import UserRepository
from recommendation_service.schemas.api import AuthResponse, TelegramAuthRequest
from recommendation_service.services.auth import AuthService


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/telegram", response_model=AuthResponse)
def telegram_auth(
    request: TelegramAuthRequest,
    db: Session = Depends(get_database),
    config: Settings = Depends(settings),
) -> AuthResponse:
    service = AuthService(
        UserRepository(db),
        config.telegram_bot_token,
        config.jwt_secret,
        config.jwt_ttl_seconds,
        config.telegram_auth_max_age_seconds,
    )
    return service.authenticate_telegram(request.init_data)

