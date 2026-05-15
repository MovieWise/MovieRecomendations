from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from recommendation_service.api.deps import current_user, get_catalog, get_database, get_inference
from recommendation_service.clients.omdb import OmdbClient
from recommendation_service.core.config import get_settings
from recommendation_service.infrastructure.database import TelegramUser
from recommendation_service.ml.hybrid import HybridInferenceService, ModelUnavailableError
from recommendation_service.repositories.omdb_cache import OmdbCacheRepository
from recommendation_service.repositories.reactions import ReactionRepository
from recommendation_service.schemas.api import RecommendationRequest, RecommendationResponse
from recommendation_service.services.catalog import MovieCatalogService
from recommendation_service.services.omdb_service import OmdbService
from recommendation_service.services.recommendations import RecommendationService


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/generate", response_model=RecommendationResponse)
async def generate_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_database),
    user: TelegramUser = Depends(current_user),
    catalog: MovieCatalogService = Depends(get_catalog),
    inference: HybridInferenceService = Depends(get_inference),
) -> RecommendationResponse:
    config = get_settings()
    omdb = OmdbService(
        OmdbClient(config.omdb_api_key, config.omdb_base_url),
        OmdbCacheRepository(db),
        catalog,
        config.omdb_cache_ttl_seconds,
    )
    service = RecommendationService(db, ReactionRepository(db), catalog, omdb, inference)
    try:
        movies, processing_time = await service.generate(user.id, request.top_n, request.candidate_top_k)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_unavailable", "missing_artifacts": exc.missing_artifacts},
        ) from exc
    return RecommendationResponse(success=True, recommendations=movies, processing_time=processing_time)

