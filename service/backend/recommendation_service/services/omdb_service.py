from __future__ import annotations

from datetime import datetime, timedelta, timezone

from recommendation_service.clients.omdb import OmdbClient
from recommendation_service.repositories.omdb_cache import OmdbCacheRepository
from recommendation_service.schemas.api import MovieInfo
from recommendation_service.services.catalog import MovieCatalogService


def normalize_omdb_payload(movie_id: int, imdb_id: str | None, payload: dict | None, fallback: MovieInfo) -> dict:
    if not payload:
        return fallback.model_dump()
    imdb = payload.get("imdbID") or imdb_id
    return {
        "movie_id": movie_id,
        "imdb_id": imdb,
        "title": payload.get("Title"),
        "year": payload.get("Year"),
        "genre": payload.get("Genre"),
        "plot": payload.get("Plot"),
        "poster": None if payload.get("Poster") == "N/A" else payload.get("Poster"),
        "rating": payload.get("imdbRating"),
        "runtime": payload.get("Runtime"),
        "director": payload.get("Director"),
        "actors": payload.get("Actors"),
        "imdb_url": f"https://www.imdb.com/title/{imdb}/" if imdb else None,
    }


class OmdbService:
    def __init__(
        self,
        client: OmdbClient,
        cache: OmdbCacheRepository,
        catalog: MovieCatalogService,
        ttl_seconds: int,
    ) -> None:
        self.client = client
        self.cache = cache
        self.catalog = catalog
        self.ttl_seconds = ttl_seconds

    async def get_movie_info(self, movie_id: int) -> MovieInfo:
        cached = self.cache.get_fresh(movie_id)
        if cached is not None and (cached.raw_payload is not None or not self.client.api_key):
            return MovieInfo(**cached.normalized_payload)

        fallback = self.catalog.get_base_info(movie_id)
        imdb_id = fallback.imdb_id
        raw = await self.client.fetch_by_imdb_id(imdb_id) if imdb_id else None
        normalized = normalize_omdb_payload(movie_id, imdb_id, raw, fallback)
        if raw is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
            self.cache.upsert(movie_id, imdb_id, normalized, raw, expires_at)
        return MovieInfo(**normalized)
