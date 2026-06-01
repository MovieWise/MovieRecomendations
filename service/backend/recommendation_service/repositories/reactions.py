from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from recommendation_service.infrastructure.database import MovieReaction


class ReactionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_for_user(self, user_id: int) -> list[MovieReaction]:
        return self.db.execute(
            select(MovieReaction)
            .where(MovieReaction.user_id == user_id)
            .order_by(MovieReaction.updated_at.desc())
        ).scalars().all()

    def get_rated_movie_ids(self, user_id: int) -> set[int]:
        rows = self.db.execute(
            select(MovieReaction.movie_id).where(MovieReaction.user_id == user_id)
        ).all()
        return {int(row[0]) for row in rows}

    def split_profile(self, user_id: int) -> tuple[list[int], list[int]]:
        reactions = self.list_for_user(user_id)
        liked = [reaction.movie_id for reaction in reactions if reaction.reaction == "like"]
        disliked = [reaction.movie_id for reaction in reactions if reaction.reaction == "dislike"]
        return liked, disliked

    def upsert(
        self,
        user_id: int,
        movie_id: int,
        reaction: str,
        source: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> MovieReaction:
        existing = self.db.execute(
            select(MovieReaction).where(
                MovieReaction.user_id == user_id,
                MovieReaction.movie_id == movie_id,
            )
        ).scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if existing is None:
            existing = MovieReaction(
                user_id=user_id,
                movie_id=movie_id,
                created_at=now,
            )
            self.db.add(existing)
        existing.reaction = reaction
        existing.source = source
        existing.session_id = session_id
        existing.metadata_json = metadata
        existing.updated_at = now
        self.db.commit()
        self.db.refresh(existing)
        return existing

    def delete_for_user(self, user_id: int, movie_id: int) -> bool:
        result = self.db.execute(
            delete(MovieReaction).where(
                MovieReaction.user_id == user_id,
                MovieReaction.movie_id == movie_id,
            )
        )
        self.db.commit()
        return bool(result.rowcount)
