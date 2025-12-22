from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import time
from sqlalchemy import select, func
from sqlalchemy.orm import Session
import numpy as np
from fastapi import Header
import os
from dotenv import load_dotenv
from sqlalchemy import delete
import secrets

load_dotenv()

# Импортируем наши модули
import infrastructure.database as db
from core.model_manager import ModelManager

# --- Pydantic модели для запросов/ответов ---
class ForwardRequest(BaseModel):
    user_id: int
    top_n: Optional[int] = 10

class ForwardResponse(BaseModel):
    success: bool
    recommendations: List[int]
    processing_time: float

# Модель для ответа истории запросов
class HistoryResponse(BaseModel):
    id: int
    user_id: int
    model_name: str
    top_n: int
    recommendations: List[int]
    processing_time: float
    timestamp: Optional[datetime] = None
    success: int
    # Этот класс позволит читать данные из ORM-объектов
    class Config:
        from_attributes = True  # Ранее назывался orm_mode

# Модель для статистики
class StatsResponse(BaseModel):
    # Общая статистика
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Время обработки
    mean_processing_time: float
    p50_processing_time: float  # 50% перцентиль (медиана)
    p95_processing_time: float  # 95% перцентиль
    p99_processing_time: float  # 99% перцентиль
    
    # Характеристики запросов (пока простые)
    unique_users_count: int
    most_common_top_n: int
    requests_per_model: dict

# --- Инициализация ModelManager ---
model_manager = ModelManager()

# Создаем функцию жизненного цикла (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Этот код выполняется ПРИ СТАРТЕ приложения
    db.init_db()
    model_manager.load_mostpop()
    print("✅ База данных и модель MostPop загружены")
    yield  # Здесь FastAPI запускается и работает
    # Этот код выполняется ПРИ ОСТАНОВКЕ (пока оставляем пустым)

# Передаем lifespan при создании приложения
app = FastAPI(title="Recommendation Service", lifespan=lifespan)


# --- Зависимость для получения сессии БД ---
def get_database():
    database = db.get_db()
    session = next(database)
    try:
        yield session
    finally:
        session.close()


def verify_delete_token(x_confirm_token: str = Header(..., description="Токен для подтверждения удаления")):
    """
    Проверяет токен для удаления истории.
    Токен берётся из переменной окружения DELETE_HISTORY_TOKEN
    """
    # Получаем токен из .env файла
    correct_token = os.getenv("DELETE_HISTORY_TOKEN")
    
    # Если в .env нет токена - ошибка конфигурации
    if not correct_token:
        raise HTTPException(
            status_code=500,
            detail="Сервер не настроен: отсутствует DELETE_HISTORY_TOKEN"
        )
    
    if not x_confirm_token:
        raise HTTPException(
            status_code=401,
            detail="Требуется токен подтверждения"
        )
    
    # Безопасное сравнение токенов
    if not secrets.compare_digest(x_confirm_token, correct_token):
        raise HTTPException(
            status_code=401,
            detail="Неверный токен подтверждения"
        )
    
    return True


# --- Эндпоинты ---
@app.get("/")
def read_root():
    return {"message": "Recommendation Service is running! Use /forward POST"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/forward", response_model=ForwardResponse)
def forward(
    request: ForwardRequest,
    db_session = Depends(get_database)
):
    """
    Получить рекомендации для пользователя.
    Пример запроса: {"user_id": 28, "top_n": 5}
    """
    start_time = time.time()
    
    try:
        # 1. Получаем рекомендации от модели
        recommendations = model_manager.predict_mostpop(
            user_id=request.user_id,
            top_n=request.top_n
        )
        
        # 2. Сохраняем запрос в историю
        history_record = db.RequestHistory(
            user_id=request.user_id,
            model_name="mostpop",
            top_n=request.top_n,
            recommendations=recommendations,
            processing_time=time.time() - start_time,
            success=1
        )
        db_session.add(history_record)
        db_session.commit()
        
        # 3. Возвращаем результат
        return ForwardResponse(
            success=True,
            recommendations=recommendations,
            processing_time=time.time() - start_time
        )
        
    except ValueError as e:
        # Пользователь не найден
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Любая другая ошибка
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    

@app.get("/history", response_model=List[HistoryResponse])
def get_history(
    skip: int = 0,
    limit: int = 100,
    db_session = Depends(get_database)
):
    """
    Получить историю запросов:
    - skip: сколько записей пропустить (для пагинации/удобства навигации)
    - limit: сколько записей вернуть (максимум)
    """
    # Получаем записи с пагинацией
    # order_by - сортируем по времени (сначала новые)
    query = select(db.RequestHistory).order_by(
        db.RequestHistory.timestamp.desc()
    ).offset(skip).limit(limit)

    # выполняем запрос
    history_records = db_session.execute(query).scalars().all()

    return history_records

@app.get("/stats", response_model= StatsResponse)
def get_stats(db_session = Depends(get_database)):
    """
    Получить статистику запросов.
    """
    # Получаем все записи истории
    query = select(db.RequestHistory)
    all_records = db_session.execute(query).scalars().all()

    if not all_records:
            # Если нет записов, возвращаем пустую статистику
            return StatsResponse(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                mean_processing_time=0,
                p50_processing_time=0,
                p95_processing_time=0,
                p99_processing_time=0,
                unique_users_count=0,
                most_common_top_n=10,
                requests_per_model={}
            )
    
    total = len(all_records)
    successful = sum(1 for r in all_records if r.success == 1)
    failed = total - successful

    processing_times = [r.processing_time for r in all_records]
    
    # Преобразуем в numpy массив для вычисления квантилей
    times_array = np.array(processing_times)
    
    # Статистика по пользователям и параметрам
    unique_users = len(set(r.user_id for r in all_records))

    # Самый популярный top_n  - мода
    top_n_values = [r.top_n for r in all_records]
    most_common_top_n = max(set(top_n_values), key=top_n_values.count)

    # ЗАПРОСЫ по моделям
    model_counts = {}
    for r in all_records:
        model_counts[r.model_name] = model_counts.get(r.model_name, 0) + 1
    
    return StatsResponse(
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        mean_processing_time=float(np.mean(times_array)),
        p50_processing_time=float(np.percentile(times_array, 50)),
        p95_processing_time=float(np.percentile(times_array, 95)),
        p99_processing_time=float(np.percentile(times_array, 99)),
        unique_users_count=unique_users,
        most_common_top_n=most_common_top_n,
        requests_per_model=model_counts
    )   

@app.delete("/history")
def delete_history(
    token_verified: bool = Depends(verify_delete_token),  # 1. Проверяем токен
    db_session = Depends(get_database)                    # 2. Получаем сессию БД
):
    """
    Удаляет всю историю запросов.
    Требует заголовок X-Confirm-Token с правильным значением.
    """

    try:
        # Создаем команду DELETE для всей таблицы
        delete_comand = delete(db.RequestHistory)

        # выполняем удаление
        result = db_session.execute(delete_comand)

        # Подтверждение
        db_session.commit()

        # возвращаем ответ
        return {
            "success": True,
            "message": f"История запросов удалена. Удалено записей: {result.rowcount}"
        }
    except Exception as e:
        # Если ошибка - откатываем изменения
        db_session.rollback()
        
        # Логируем ошибку (в консоль)
        print(f"Ошибка при удалении истории: {e}")
        
        # Возвращаем ошибку пользователю
        raise HTTPException(
            status_code=500,
            detail="Не удалось удалить историю запросов"
        )