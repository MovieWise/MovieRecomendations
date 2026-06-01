from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from recommendation_service.api.deps import get_database
from recommendation_service.api.routers import auth, health, movies, reactions, recommendations
from recommendation_service.core.config import get_settings
from recommendation_service.core.model_manager import ModelManager
from recommendation_service.infrastructure import database as db
from recommendation_service.ml.hybrid import HybridArtifacts, HybridInferenceService
from recommendation_service.services.catalog import MovieCatalogService


class ForwardRequest(BaseModel):
    user_id: int
    model: str = "mostpop"
    top_n: Optional[int] = 10


class ForwardResponse(BaseModel):
    success: bool
    recommendations: list[int]
    processing_time: float


class PredictRawRequest(BaseModel):
    selected_movie_ids: list[int]
    model: str = "puresvd"
    top_n: int = 5


class HistoryResponse(BaseModel):
    id: int
    user_id: int
    model_name: str
    top_n: int
    recommendations: list[int]
    processing_time: float
    timestamp: Optional[datetime] = None
    success: int

    class Config:
        from_attributes = True


legacy_model_manager = ModelManager()
legacy_loaded_models: set[str] = set()


def _load_legacy_models() -> None:
    for name, loader in {
        "mostpop": legacy_model_manager.load_mostpop,
        "puresvd": legacy_model_manager.load_puresvd,
        "ease": legacy_model_manager.load_ease,
    }.items():
        try:
            loader()
            legacy_loaded_models.add(name)
        except FileNotFoundError:
            continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_settings()
    db.init_db(config.database_url)
    app.state.catalog = MovieCatalogService(config.movies_path, config.links_path, config.content_features_path)
    app.state.inference = HybridInferenceService(
        HybridArtifacts(
            ease_weights_path=config.ease_weights_path,
            ease_item_encoder_path=config.ease_item_encoder_path,
            ease_user_encoder_path=config.ease_user_encoder_path,
            ease_interactions_path=config.ease_interactions_path,
            lgbm_ranker_path=config.lgbm_ranker_path,
        )
    )
    _load_legacy_models()
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = get_settings().api_prefix
app.include_router(auth.router, prefix=api_prefix)
app.include_router(movies.router, prefix=api_prefix)
app.include_router(reactions.router, prefix=api_prefix)
app.include_router(recommendations.router, prefix=api_prefix)
app.include_router(health.router, prefix=api_prefix)


@app.get("/")
def read_root():
    return {"message": "Recommendation Service is running", "api": api_prefix}


@app.post("/forward", response_model=ForwardResponse)
def forward(request: ForwardRequest, db_session: Session = Depends(get_database)):
    if request.model not in ["mostpop", "puresvd", "ease"]:
        raise HTTPException(status_code=400, detail="unsupported_model")
    if request.model not in legacy_loaded_models:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "model": request.model})

    start_time = time.time()
    recommendations: list[int] = []
    success = 0
    error: str | None = None
    try:
        if request.model == "mostpop":
            recommendations = legacy_model_manager.predict_mostpop(request.user_id, request.top_n)
        elif request.model == "puresvd":
            recommendations = legacy_model_manager.predict_puresvd(request.user_id, request.top_n)
        else:
            recommendations = legacy_model_manager.predict_ease(request.user_id, request.top_n)
        success = 1
        return ForwardResponse(success=True, recommendations=recommendations, processing_time=time.time() - start_time)
    except ValueError as exc:
        error = str(exc)
        raise HTTPException(status_code=400, detail=error) from exc
    finally:
        db_session.add(
            db.RequestHistory(
                user_id=request.user_id,
                model_name=request.model,
                top_n=request.top_n,
                recommendations=recommendations,
                processing_time=time.time() - start_time,
                success=success,
                error=error,
            )
        )
        db_session.commit()


@app.post("/predict_raw", response_model=ForwardResponse)
def predict_raw(request: PredictRawRequest):
    if request.model not in ["puresvd", "ease"]:
        raise HTTPException(status_code=400, detail="unsupported_model")
    if request.model not in legacy_loaded_models:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "model": request.model})
    start_time = time.time()
    recommendations = legacy_model_manager.predict_for_new_user(request.selected_movie_ids, request.model, request.top_n)
    return ForwardResponse(success=True, recommendations=recommendations, processing_time=time.time() - start_time)


@app.get("/history", response_model=list[HistoryResponse])
def get_history(skip: int = 0, limit: int = 100, db_session: Session = Depends(get_database)):
    return (
        db_session.query(db.RequestHistory)
        .order_by(db.RequestHistory.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

