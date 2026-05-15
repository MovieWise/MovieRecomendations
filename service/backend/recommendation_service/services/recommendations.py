from __future__ import annotations

import time

from sqlalchemy.orm import Session

from recommendation_service.infrastructure.database import RequestHistory
from recommendation_service.ml.hybrid import HybridInferenceService, ModelUnavailableError
from recommendation_service.repositories.reactions import ReactionRepository
from recommendation_service.schemas.api import MovieInfo
from recommendation_service.services.catalog import MovieCatalogService
from recommendation_service.services.omdb_service import OmdbService


class RecommendationService:
    def __init__(
        self,
        db: Session,
        reactions: ReactionRepository,
        catalog: MovieCatalogService,
        omdb: OmdbService,
        inference: HybridInferenceService,
    ) -> None:
        self.db = db
        self.reactions = reactions
        self.catalog = catalog
        self.omdb = omdb
        self.inference = inference

    async def generate(self, user_id: int, top_n: int, candidate_top_k: int) -> tuple[list[MovieInfo], float]:
        start = time.time()
        success = 0
        error: str | None = None
        recommendations: list[int] = []
        try:
            liked, disliked = self.reactions.split_profile(user_id)
            recommendations = self.inference.recommend(
                liked_movie_ids=liked,
                disliked_movie_ids=disliked,
                feature_frame=self.catalog.content,
                top_n=top_n,
                candidate_top_k=candidate_top_k,
            )
            success = 1
            enriched = [await self.omdb.get_movie_info(movie_id) for movie_id in recommendations]
            return enriched, time.time() - start
        except ModelUnavailableError as exc:
            error = "model_unavailable:" + ",".join(exc.missing_artifacts)
            raise
        finally:
            processing_time = time.time() - start
            self.db.add(
                RequestHistory(
                    user_id=user_id,
                    model_name="ease_lgbm",
                    top_n=top_n,
                    recommendations=recommendations,
                    processing_time=processing_time,
                    success=success,
                    error=error,
                )
            )
            self.db.commit()

