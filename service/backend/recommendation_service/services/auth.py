from __future__ import annotations

from recommendation_service.core.security import create_access_token, validate_telegram_init_data
from recommendation_service.repositories.users import UserRepository
from recommendation_service.schemas.api import AuthResponse, UserResponse


class AuthService:
    def __init__(
        self,
        users: UserRepository,
        telegram_bot_token: str,
        jwt_secret: str,
        jwt_ttl_seconds: int,
        telegram_auth_max_age_seconds: int,
    ) -> None:
        self.users = users
        self.telegram_bot_token = telegram_bot_token
        self.jwt_secret = jwt_secret
        self.jwt_ttl_seconds = jwt_ttl_seconds
        self.telegram_auth_max_age_seconds = telegram_auth_max_age_seconds

    def authenticate_telegram(self, init_data: str) -> AuthResponse:
        telegram_user = validate_telegram_init_data(
            init_data,
            self.telegram_bot_token,
            max_age_seconds=self.telegram_auth_max_age_seconds,
        )
        user = self.users.upsert_from_telegram(telegram_user)
        token = create_access_token({"sub": str(user.id), "telegram_id": user.telegram_id}, self.jwt_secret, self.jwt_ttl_seconds)
        return AuthResponse(
            access_token=token,
            user=UserResponse(
                id=user.id,
                telegram_id=user.telegram_id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            ),
        )

