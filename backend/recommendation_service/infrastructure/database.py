from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

Base = declarative_base()

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

# SQLite файл будет в data/database.db
engine = create_engine('sqlite:///./data/database.db')
SessionLocal = sessionmaker(bind = engine)

def init_db():
    """Создает таблицы"""
    Base.metadata.create_all(engine)
    print('Таблицы созданы!')

def get_db():
    """Для получения сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

        