"""Baseline recommenders migrated from the notebooks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recs.utils.io import load_pickle, save_pickle


class RandomRecommender:
    """Random item sampler for sanity checks."""

    def __init__(self) -> None:
        self.items: list[int] = []
        self.trained = False

    def fit(self, train_data: pd.DataFrame, col: str = "train_interactions") -> "RandomRecommender":
        items: set[int] = set()
        for interactions in train_data[col]:
            items.update(item for item, _ in interactions)
        self.items = sorted(items)
        self.trained = True
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if not self.trained:
            raise RuntimeError("Model is not fitted.")
        top_k = min(top_k, len(self.items))
        return np.random.choice(self.items, replace=False, size=top_k).tolist()

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        return self.recommend(user_id=-1, top_k=top_k)

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomRecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model


class TopPopularRecommender:
    """Top-pop recommender based on interaction counts."""

    def __init__(self) -> None:
        self.recommendations: list[int] = []
        self.trained = False

    def fit(self, train_data: pd.DataFrame, col: str = "train_interactions") -> "TopPopularRecommender":
        counts: Counter[int] = Counter()
        for interactions in train_data[col]:
            counts.update(item for item, _ in interactions)
        self.recommendations = [item for item, _ in counts.most_common()]
        self.trained = True
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if not self.trained:
            raise RuntimeError("Model is not fitted.")
        return self.recommendations[:top_k]

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        return self.recommend(user_id=-1, top_k=top_k)

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "TopPopularRecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model


class ModifiedTopPopularRecommender(TopPopularRecommender):
    """Top-pop recommender that filters seen items for known users."""

    def __init__(self) -> None:
        super().__init__()
        self.user_seen: dict[int, set[int]] = {}

    def fit(
        self,
        train_data: pd.DataFrame,
        user_col: str = "userId",
        interactions_col: str = "train_interactions",
    ) -> "ModifiedTopPopularRecommender":
        super().fit(train_data, col=interactions_col)
        self.user_seen = {
            int(row[user_col]): {item for item, _ in row[interactions_col]}
            for _, row in train_data.iterrows()
        }
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if not self.trained:
            raise RuntimeError("Model is not fitted.")
        seen = self.user_seen.get(int(user_id), set()) if exclude_seen else set()
        recs: list[int] = []
        for item in self.recommendations:
            if item in seen:
                continue
            recs.append(item)
            if len(recs) == top_k:
                break
        return recs

