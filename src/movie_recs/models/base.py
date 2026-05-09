"""Shared recommender interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaseRecommender(Protocol):
    """Stable interface for training and serving recommenders."""

    def fit(self, train_data: pd.DataFrame, *args, **kwargs) -> "BaseRecommender":
        """Train a model."""

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        """Generate user recommendations."""

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        """Generate recommendations from a list of preferred items."""

    def save(self, path: str | Path) -> Path:
        """Persist a trained model."""

    @classmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        """Load a persisted model."""
