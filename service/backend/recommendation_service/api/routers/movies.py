from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from recommendation_service.api.deps import current_user, get_catalog, get_database
from recommendation_service.clients.omdb import OmdbClient
from recommendation_service.core.config import get_settings
from recommendation_service.infrastructure.database import TelegramUser
from recommendation_service.repositories.omdb_cache import OmdbCacheRepository
from recommendation_service.repositories.reactions import ReactionRepository
from recommendation_service.schemas.api import MovieFeedResponse, MovieInfo
from recommendation_service.services.catalog import MovieCatalogService
from recommendation_service.services.omdb_service import OmdbService


router = APIRouter(prefix="/movies", tags=["movies"])


def _omdb_service(db: Session, catalog: MovieCatalogService) -> OmdbService:
    config = get_settings()
    return OmdbService(
        OmdbClient(config.omdb_api_key, config.omdb_base_url),
        OmdbCacheRepository(db),
        catalog,
        config.omdb_cache_ttl_seconds,
    )


@router.get("/feed", response_model=MovieFeedResponse)
async def movie_feed(
    limit: int = 20,
    db: Session = Depends(get_database),
    user: TelegramUser = Depends(current_user),
    catalog: MovieCatalogService = Depends(get_catalog),
) -> MovieFeedResponse:
    reactions = ReactionRepository(db)
    excluded = reactions.get_rated_movie_ids(user.id)
    omdb = _omdb_service(db, catalog)
    movie_ids = catalog.feed(excluded, limit=min(max(limit, 1), 50))
    movies = [await omdb.get_movie_info(movie_id) for movie_id in movie_ids]
    return MovieFeedResponse(movies=movies, rated_count=len(excluded))


@router.get("/search", response_model=MovieFeedResponse)
async def movie_search(
    q: str = "",
    limit: int = 20,
    db: Session = Depends(get_database),
    user: TelegramUser = Depends(current_user),
    catalog: MovieCatalogService = Depends(get_catalog),
) -> MovieFeedResponse:
    reactions = ReactionRepository(db)
    excluded = reactions.get_rated_movie_ids(user.id)
    omdb = _omdb_service(db, catalog)
    movie_ids = catalog.search(q, excluded_movie_ids=excluded, limit=min(max(limit, 1), 50))
    movies = [await omdb.get_movie_info(movie_id) for movie_id in movie_ids]
    return MovieFeedResponse(movies=movies, rated_count=len(excluded))


@router.get("/{movie_id}", response_model=MovieInfo)
async def movie_detail(
    movie_id: int,
    db: Session = Depends(get_database),
    _: TelegramUser = Depends(current_user),
    catalog: MovieCatalogService = Depends(get_catalog),
) -> MovieInfo:
    return await _omdb_service(db, catalog).get_movie_info(movie_id)
