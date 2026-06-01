from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from recommendation_service.core.config import Settings, get_settings
from recommendation_service.core.security import decode_access_token
from recommendation_service.infrastructure.database import TelegramUser, get_db
from recommendation_service.repositories.users import UserRepository


def get_database() -> Session:
    database = get_db()
    session = next(database)
    try:
        yield session
    finally:
        session.close()


def settings() -> Settings:
    return get_settings()


def get_catalog(request: Request):
    return request.app.state.catalog


def get_inference(request: Request):
    return request.app.state.inference


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_database),
    config: Settings = Depends(settings),
) -> TelegramUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authorization_required")
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token, config.jwt_secret)
    user_id = int(payload["sub"])
    user = UserRepository(db).get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_not_found")
    return user

