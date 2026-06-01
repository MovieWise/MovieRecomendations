from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from recommendation_service.infrastructure.database import OmdbCache


class OmdbCacheRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_fresh(self, movie_id: int) -> OmdbCache | None:
        row = self.db.execute(
            select(OmdbCache).where(OmdbCache.movie_id == movie_id)
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            return None
        return row

    def upsert(
        self,
        movie_id: int,
        imdb_id: str | None,
        normalized_payload: dict,
        raw_payload: dict | None,
        expires_at: datetime,
    ) -> OmdbCache:
        row = self.db.execute(
            select(OmdbCache).where(OmdbCache.movie_id == movie_id)
        ).scalar_one_or_none()
        if row is None:
            row = OmdbCache(movie_id=movie_id)
            self.db.add(row)
        row.imdb_id = imdb_id
        row.normalized_payload = normalized_payload
        row.raw_payload = raw_payload
        row.fetched_at = datetime.now(timezone.utc)
        row.expires_at = expires_at
        self.db.commit()
        self.db.refresh(row)
        return row

