"""Serving-friendly wrappers around the reusable models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RecommenderService:
    """Simple adapter that standardizes inference calls."""

    model: Any
    user_encoder: Any | None = None
    item_encoder: Any | None = None

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> list[int]:
        if self.user_encoder is not None and user_id in getattr(self.user_encoder, "classes_", []):
            user_id = int(self.user_encoder.transform([user_id])[0])
        recs = self.model.recommend(user_id=user_id, top_k=top_k)
        if self.item_encoder is not None:
            return self.item_encoder.inverse_transform(recs).tolist()
        return recs

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        encoded = item_ids
        if self.item_encoder is not None:
            encoded = self.item_encoder.transform(item_ids).tolist()
        recs = self.model.recommend_for_profile(encoded, top_k=top_k)
        if self.item_encoder is not None:
            return self.item_encoder.inverse_transform(recs).tolist()
        return recs

