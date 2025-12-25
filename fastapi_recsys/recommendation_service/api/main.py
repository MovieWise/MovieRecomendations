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

# Modules import
import infrastructure.database as db
from core.model_manager import ModelManager

# Pydantic models
class ForwardRequest(BaseModel):
    user_id: int
    model: str = "mostpop"
    top_n: Optional[int] = 10

class ForwardResponse(BaseModel):
    success: bool
    recommendations: List[int]
    processing_time: float

# History model
class HistoryResponse(BaseModel):
    id: int
    user_id: int
    model_name: str
    top_n: int
    recommendations: List[int]
    processing_time: float
    timestamp: Optional[datetime] = None
    success: int
    # Class for reading data from orm objects
    class Config:
        from_attributes = True

# Statistic model
class StatsResponse(BaseModel):
    # General statistic
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Processing time
    mean_processing_time: float
    p50_processing_time: float  # 50% percentile
    p95_processing_time: float  # 95% percentile
    p99_processing_time: float  # 99% percentile
    
    # Query characteristics
    unique_users_count: int
    most_common_top_n: int
    requests_per_model: dict

# ModelManager initialization
model_manager = ModelManager()

# Lifespan creation
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load database
    db.init_db()

    # Load models
    try:
        model_manager.load_mostpop()
    except FileNotFoundError:
        print("Ошибка при загрузке MostPop - Файл не найден!")
    try:
        model_manager.load_puresvd()
    except FileNotFoundError:
        print("Ошибка при загрузке PureSVD - Файл не найден!") 
    try:
        model_manager.load_ease()
    except FileNotFoundError:
        print("Ошибка при загрузке EASE - Файл не найден!") 

    print("База данных и модели загружены")
    yield


app = FastAPI(title="Recommendation Service", lifespan=lifespan)


# Dependency for getting a DB session
def get_database():
    database = db.get_db()
    session = next(database)
    try:
        yield session
    finally:
        session.close()


def verify_delete_token(x_confirm_token: str = Header(..., description="Токен для подтверждения удаления")):
    """
    Checks the token for deleting history.
    The token is taken from the DELETE_HISTORY_TOKEN environment variable
    """
    # Geting token from .env file
    correct_token = os.getenv("DELETE_HISTORY_TOKEN")
    
    # if not - error
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
    
    # Safe token comparison
    if not secrets.compare_digest(x_confirm_token, correct_token):
        raise HTTPException(
            status_code=401,
            detail="Неверный токен подтверждения"
        )
    
    return True


# Endpoints
@app.get("/")
def read_root():
    return {"message": "Recommendation Service is running! Use /forward POST"}

@app.post("/forward", response_model=ForwardResponse)
def forward(
    request: ForwardRequest,
    db_session = Depends(get_database)
):
    """
    Get user recomendations
    Request example: {"user_id": 28, "top_n": 5}
    """
    if request.model not in ["mostpop", "puresvd", "ease"]:
        raise HTTPException(
            status_code=400,
            detail=f"Bad Request: Модель '{request.model}' не поддерживается. Доступно: mostpop, puresvd, ease"
        )
    start_time = time.time()
    recommendations = []
    success = 0

    try:
        if request.model == "mostpop":
            # Get recs from model
            recommendations = model_manager.predict_mostpop(
                user_id=request.user_id,
                top_n=request.top_n
            )
        elif request.model == "puresvd":
            recommendations = model_manager.predict_puresvd(
                user_id=request.user_id,
                top_n=request.top_n
            )
        elif request.model == "ease":
            recommendations = model_manager.predict_ease(
                user_id=request.user_id,
                top_n=request.top_n
            )
        
        success = 1

    except ValueError as e:
        # User does not exist
        error_detail = str(e)
        raise_type = HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        # Other error
        error_detail = "модель не смогла обработать данные"
        raise_type = HTTPException(status_code=403, detail=error_detail)

    finally:
        processing_time=time.time() - start_time
        # Save request in history
        history_record = db.RequestHistory(
            user_id=request.user_id,
            model_name=request.model,
            top_n=request.top_n,
            recommendations=recommendations,
            processing_time=processing_time,
            success=success
        )
        db_session.add(history_record)
        db_session.commit()

        if success == 0:
            raise raise_type
        
        # Return result
        return ForwardResponse(
            success=True,
            recommendations=recommendations,
            processing_time=processing_time
        )
    

@app.get("/history", response_model=List[HistoryResponse])
def get_history(
    skip: int = 0,
    limit: int = 100,
    db_session = Depends(get_database)
):
    """
    Get requests history:
    - skip: how many records to skip
    - limit: how many records to return
    """

    # Getting records with pagination
    query = select(db.RequestHistory).order_by(
        db.RequestHistory.timestamp.desc()
    ).offset(skip).limit(limit)

    # Execute the request
    history_records = db_session.execute(query).scalars().all()

    return history_records

@app.get("/stats", response_model= StatsResponse)
def get_stats(db_session = Depends(get_database)):
    """
    Get requests statistic
    """
    # Get all requests
    query = select(db.RequestHistory)
    all_records = db_session.execute(query).scalars().all()

    if not all_records:
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
    
    # Converting a Numpy array
    times_array = np.array(processing_times)
    
    # Statistics by users and parameters
    unique_users = len(set(r.user_id for r in all_records))

    # The most popular top_n - mode
    top_n_values = [r.top_n for r in all_records]
    most_common_top_n = max(set(top_n_values), key=top_n_values.count)

    # Request by models
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
    token_verified: bool = Depends(verify_delete_token),  # Checking the token
    db_session = Depends(get_database)                    # Getting a db session
):
    """
    Deletes all query history.
    Requires an X-Confirm-Token header with a valid value.
    """

    try:

        delete_comand = delete(db.RequestHistory)

        result = db_session.execute(delete_comand)

        db_session.commit()

        return {
            "success": True,
            "message": f"История запросов удалена. Удалено записей: {result.rowcount}"
        }
    except Exception as e:

        db_session.rollback()
        
        print(f"Ошибка при удалении истории: {e}")
        
        # Returning an error
        raise HTTPException(
            status_code=500,
            detail="Не удалось удалить историю запросов"
        )
