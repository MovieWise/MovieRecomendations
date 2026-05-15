from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from recommendation_service.api.deps import get_database, get_inference, settings
from recommendation_service.core.config import Settings
from recommendation_service.ml.hybrid import HybridInferenceService
from recommendation_service.schemas.api import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck(
    db: Session = Depends(get_database),
    config: Settings = Depends(settings),
    inference: HybridInferenceService = Depends(get_inference),
) -> HealthResponse:
    database_status = "ok"
    try:
        db.execute(text("select 1"))
    except Exception:
        database_status = "error"
    model_available = inference.available
    return HealthResponse(
        status="ok" if database_status == "ok" else "degraded",
        database=database_status,
        omdb_configured=bool(config.omdb_api_key),
        model_available=model_available,
        model_missing_artifacts=inference.missing_artifacts,
    )

