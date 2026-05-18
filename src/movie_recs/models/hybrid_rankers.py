"""Hybrid ranking models and candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movie_recs.preprocessing.ranking_dataset import DEFAULT_RANKING_FEATURES
from movie_recs.utils.io import load_pickle, save_pickle


def get_ease_candidates(
    user_items_list: list[tuple[int, float]],
    item_encoder: Any,
    weights: np.ndarray,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Generate EASE candidates exactly like in the hybrid notebook."""
    user_items = [item_id for item_id, _ in user_items_list]
    known_items = [item_id for item_id in user_items if item_id in item_encoder.classes_]
    if not known_items:
        return []
    encoded = item_encoder.transform(known_items)
    vector = np.zeros(weights.shape[0], dtype=float)
    vector[encoded] = 1.0
    preds = np.asarray(vector @ weights).flatten()
    preds[encoded] = -1e9
    top_k = min(top_k, len(preds))
    top_indices = np.argsort(-preds)[:top_k]
    top_scores = preds[top_indices]
    decoded = item_encoder.inverse_transform(top_indices)
    return list(zip(decoded.tolist(), top_scores.tolist()))


@dataclass(slots=True)
class LightGBMHybridRanker:
    """Thin wrapper around LightGBM ranking/classification models."""

    feature_names: list[str]
    model: Any | None = None
    categorical_features: list[str] | None = None

    def fit(self, train_frame: pd.DataFrame, label_col: str = "label", group: list[int] | None = None) -> "LightGBMHybridRanker":
        if self.model is None:
            raise ValueError("Provide an instantiated LightGBM model.")
        self.model.fit(
            X=train_frame[self.feature_names],
            y=train_frame[label_col],
            group=group,
        )
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(frame[self.feature_names])[:, 1]
        return self.model.predict(frame[self.feature_names])

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMHybridRanker":
        model = cls(feature_names=list(DEFAULT_RANKING_FEATURES))
        model.__dict__.update(load_pickle(path))
        return model


@dataclass(slots=True)
class CatBoostHybridRanker:
    """Thin wrapper around CatBoost rankers."""

    feature_names: list[str]
    model: Any | None = None

    def fit(self, train_pool: Any) -> "CatBoostHybridRanker":
        if self.model is None:
            raise ValueError("Provide an instantiated CatBoost model.")
        self.model.fit(train_pool)
        return self

    def predict_scores(self, pool: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict(pool)

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostHybridRanker":
        model = cls(feature_names=list(DEFAULT_RANKING_FEATURES))
        model.__dict__.update(load_pickle(path))
        return model
