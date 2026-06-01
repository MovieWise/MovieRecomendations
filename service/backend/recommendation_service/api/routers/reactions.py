from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from recommendation_service.api.deps import current_user, get_catalog, get_database
from recommendation_service.clients.omdb import OmdbClient
from recommendation_service.core.config import get_settings
from recommendation_service.infrastructure.database import TelegramUser
from recommendation_service.repositories.omdb_cache import OmdbCacheRepository
from recommendation_service.repositories.reactions import ReactionRepository
from recommendation_service.schemas.api import (
    ProfileReaction,
    ProfileResponse,
    ReactionRequest,
    ReactionResponse,
    UserResponse,
)
from recommendation_service.services.catalog import MovieCatalogService
from recommendation_service.services.omdb_service import OmdbService


router = APIRouter(tags=["reactions"])


@router.post("/reactions", response_model=ReactionResponse)
def save_reaction(
    request: ReactionRequest,
    db: Session = Depends(get_database),
    user: TelegramUser = Depends(current_user),
) -> ReactionResponse:
    reaction = ReactionRepository(db).upsert(
        user_id=user.id,
        movie_id=request.movie_id,
        reaction=request.reaction,
        source=request.source,
        session_id=request.session_id,
        metadata=request.metadata,
    )
    return ReactionResponse(movie_id=reaction.movie_id, reaction=reaction.reaction, updated_at=reaction.updated_at)


@router.delete("/reactions/{movie_id}")
def delete_reaction(
    movie_id: int,
    db: Session = Depends(get_database),
    user: TelegramUser = Depends(current_user),
) -> dict[str, bool | int]:
    deleted = ReactionRepository(db).delete_for_user(user.id, movie_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="reaction_not_found")
    return {"success": True, "movie_id": movie_id}


@router.get("/profile/ratings", response_model=ProfileResponse)
async def profile_ratings(
    db: Session = Depends(get_database),
    user: TelegramUser = Depends(current_user),
    catalog: MovieCatalogService = Depends(get_catalog),
) -> ProfileResponse:
    ratings = ReactionRepository(db).list_for_user(user.id)
    config = get_settings()
    omdb = OmdbService(
        OmdbClient(config.omdb_api_key, config.omdb_base_url),
        OmdbCacheRepository(db),
        catalog,
        config.omdb_cache_ttl_seconds,
    )
    profile = [
        ProfileReaction(
            movie_id=row.movie_id,
            reaction=row.reaction,
            updated_at=row.updated_at,
            movie=await omdb.get_movie_info(row.movie_id),
        )
        for row in ratings
    ]
    return ProfileResponse(
        user=UserResponse(
            id=user.id,
            telegram_id=user.telegram_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        ),
        ratings=profile,
        liked_count=sum(1 for row in ratings if row.reaction == "like"),
        disliked_count=sum(1 for row in ratings if row.reaction == "dislike"),
    )
