from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from recommendation_service.infrastructure.database import TelegramUser


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, user_id: int) -> TelegramUser | None:
        return self.db.get(TelegramUser, user_id)

    def get_by_telegram_id(self, telegram_id: int) -> TelegramUser | None:
        return self.db.execute(
            select(TelegramUser).where(TelegramUser.telegram_id == telegram_id)
        ).scalar_one_or_none()

    def upsert_from_telegram(self, telegram_user: dict) -> TelegramUser:
        telegram_id = int(telegram_user["id"])
        user = self.get_by_telegram_id(telegram_id)
        if user is None:
            user = TelegramUser(telegram_id=telegram_id)
            self.db.add(user)
        user.username = telegram_user.get("username")
        user.first_name = telegram_user.get("first_name")
        user.last_name = telegram_user.get("last_name")
        user.language_code = telegram_user.get("language_code")
        self.db.commit()
        self.db.refresh(user)
        return user

