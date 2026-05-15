from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from typing import Generator

from recommendation_service.core.config import get_settings

Base = declarative_base()

class TelegramUser(Base):
    __tablename__ = "telegram_users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    language_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class MovieReaction(Base):
    __tablename__ = "movie_reactions"
    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_user_movie_reaction"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("telegram_users.id"), index=True, nullable=False)
    movie_id = Column(Integer, index=True, nullable=False)
    reaction = Column(String, nullable=False)
    source = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class OmdbCache(Base):
    __tablename__ = "omdb_cache"

    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, unique=True, index=True, nullable=False)
    imdb_id = Column(String, index=True, nullable=True)
    normalized_payload = Column(JSON, nullable=False)
    raw_payload = Column(JSON, nullable=True)
    fetched_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime, nullable=False)


class RequestHistory(Base):
    __tablename__ = 'requests_history'

    id = Column(Integer, primary_key = True)
    user_id = Column(Integer)
    model_name = Column(String)
    top_n = Column(Integer)
    recommendations = Column(JSON)
    processing_time = Column(Float)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    success = Column(Integer)
    error = Column(String, nullable=True)


engine = None
SessionLocal = sessionmaker()


def configure_database(database_url: str | None = None):
    global engine
    target_url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if target_url.startswith("sqlite") else {}
    engine = create_engine(target_url, connect_args=connect_args)
    SessionLocal.configure(bind=engine)
    return engine

def init_db(database_url: str | None = None):
    """Создает таблицы"""
    active_engine = configure_database(database_url) if database_url else engine or configure_database()
    Base.metadata.create_all(active_engine)
    print('Таблицы созданы!')

def get_db() -> Generator:
    """Для получения сессии БД"""
    if engine is None:
        configure_database()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

        
