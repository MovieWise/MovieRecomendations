"""Hybrid and reranking dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from movie_recs.preprocessing.content import parse_list_string


DEFAULT_RANKING_FEATURES = [
    "score",
    "popularity",
    "age",
    "rating",
    "runtimeMinutes",
    "num_translations",
    "cluster",
    "is_multiregional",
    "has_director",
    "num_interactions",
    "main_genre",
    "main_region",
    "main_actor_popularity",
]


@dataclass(slots=True)
class RankingDatasetBuilder:
    """Create hybrid model training tables from candidate lists and metadata."""

    feature_names: Sequence[str] = tuple(DEFAULT_RANKING_FEATURES)

    def enrich_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values and derive ranking features."""
        result = frame.copy()
        for col in ["age", "runtimeMinutes", "num_translations"]:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].median())
        if "cluster" in result.columns:
            result["cluster"] = result["cluster"].fillna(result["cluster"].mode().iloc[0])
        if "main_region" in result.columns:
            result["main_region"] = result["main_region"].fillna("UNKNOWN")
        if "is_multiregional" in result.columns:
            result["is_multiregional"] = result["is_multiregional"].apply(lambda value: 0 if value == 0.0 else int(bool(value)))
        result["has_director"] = result.get("director_name", pd.Series(index=result.index)).notna().astype(int)
        result["main_genre"] = result["genres"].apply(parse_list_string)
        actor_first = result["actors_list"].apply(parse_list_string)
        actor_counts = actor_first.value_counts()
        result["main_actor_popularity"] = actor_first.map(lambda value: actor_counts.get(value, 0))
        return result

    def build_labels(
        self,
        frame: pd.DataFrame,
        truth_items: dict[int, set[int]],
        user_col: str = "userId",
        item_col: str = "movieId",
    ) -> pd.DataFrame:
        """Attach binary labels based on held-out positives."""
        result = frame.copy()
        result["label"] = result.apply(
            lambda row: int(row[item_col] in truth_items.get(int(row[user_col]), set())),
            axis=1,
        )
        return result

    def build(self, frame: pd.DataFrame, truth_items: dict[int, set[int]] | None = None) -> pd.DataFrame:
        """Prepare a ranking dataset end to end."""
        result = self.enrich_features(frame)
        if truth_items is not None:
            result = self.build_labels(result, truth_items)
        return result

