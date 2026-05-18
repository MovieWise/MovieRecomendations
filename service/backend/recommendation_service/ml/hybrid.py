from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from movie_recs.models.hybrid_rankers import LightGBMHybridRanker, get_ease_candidates
from movie_recs.preprocessing.ranking_dataset import DEFAULT_RANKING_FEATURES, RankingDatasetBuilder


@dataclass(frozen=True, slots=True)
class HybridArtifacts:
    ease_weights_path: str
    ease_item_encoder_path: str
    ease_user_encoder_path: str
    ease_interactions_path: str
    lgbm_ranker_path: str


class ModelUnavailableError(RuntimeError):
    def __init__(self, missing_artifacts: list[str]) -> None:
        super().__init__("model_unavailable")
        self.missing_artifacts = missing_artifacts


class HybridInferenceService:
    def __init__(self, artifacts: HybridArtifacts) -> None:
        self.artifacts = artifacts
        self.ease_weights: np.ndarray | None = None
        self.item_encoder: Any | None = None
        self.ranker: LightGBMHybridRanker | None = None
        self._missing_artifacts: list[str] = []

    @property
    def missing_artifacts(self) -> list[str]:
        self._ensure_loaded()
        return list(self._missing_artifacts)

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return not self._missing_artifacts

    def _ensure_loaded(self) -> None:
        if self.ease_weights is not None and self.item_encoder is not None and self.ranker is not None:
            return
        if self._missing_artifacts:
            return
        self.ease_weights = None
        self.item_encoder = None
        self.ranker = None
        ease_item_encoder_path = self.artifacts.ease_item_encoder_path
        legacy_item_encoder_path = str(Path(ease_item_encoder_path).with_name("item_encoder.joblib"))
        if not Path(ease_item_encoder_path).exists() and Path(legacy_item_encoder_path).exists():
            ease_item_encoder_path = legacy_item_encoder_path
        required = [
            self.artifacts.ease_weights_path,
            ease_item_encoder_path,
            self.artifacts.lgbm_ranker_path,
        ]
        self._missing_artifacts = [path for path in required if not Path(path).exists()]
        if self._missing_artifacts:
            return
        try:
            self.ease_weights = np.load(self.artifacts.ease_weights_path)
            self.item_encoder = joblib.load(ease_item_encoder_path)
            self.ranker = LightGBMHybridRanker.load(self.artifacts.lgbm_ranker_path)
            if self.ranker.model is None:
                self.ease_weights = None
                self.item_encoder = None
                self.ranker = None
                self._missing_artifacts = [f"invalid_ranker:{self.artifacts.lgbm_ranker_path}"]
        except ModuleNotFoundError as exc:
            self.ease_weights = None
            self.item_encoder = None
            self.ranker = None
            self._missing_artifacts = [f"python_dependency:{exc.name}"]
        except OSError as exc:
            self.ease_weights = None
            self.item_encoder = None
            self.ranker = None
            self._missing_artifacts = [f"native_dependency:{exc}"]

    def recommend(
        self,
        liked_movie_ids: list[int],
        disliked_movie_ids: list[int],
        feature_frame: pd.DataFrame,
        top_n: int,
        candidate_top_k: int,
    ) -> list[int]:
        self._ensure_loaded()
        if self._missing_artifacts:
            raise ModelUnavailableError(self._missing_artifacts)
        if self.ease_weights is None or self.item_encoder is None or self.ranker is None:
            raise ModelUnavailableError(["model_not_loaded"])
        if not liked_movie_ids:
            return []

        candidates = get_ease_candidates(
            [(movie_id, 1.0) for movie_id in liked_movie_ids],
            self.item_encoder,
            self.ease_weights,
            top_k=candidate_top_k,
        )
        excluded = set(liked_movie_ids) | set(disliked_movie_ids)
        candidates = [(movie_id, score) for movie_id, score in candidates if movie_id not in excluded]
        if not candidates:
            return []

        ranking_frame = self._prepare_ranking_frame(candidates, feature_frame)
        scores = self.ranker.predict_scores(ranking_frame)
        ranking_frame = ranking_frame.assign(_rank_score=scores)
        ranking_frame = ranking_frame.sort_values("_rank_score", ascending=False)
        return ranking_frame["movieId"].astype(int).head(top_n).tolist()

    def _prepare_ranking_frame(self, candidates: list[tuple[int, float]], feature_frame: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(candidates, columns=["movieId", "score"])
        if not feature_frame.empty and "movieId" in feature_frame.columns:
            frame = frame.merge(feature_frame, on="movieId", how="left", suffixes=("", "_feature"))
            if "score_feature" in frame.columns:
                frame = frame.drop(columns=["score_feature"])
        frame = self._ensure_feature_inputs(frame)
        frame = RankingDatasetBuilder().build(frame)
        for name in DEFAULT_RANKING_FEATURES:
            if name not in frame.columns:
                frame[name] = "UNKNOWN" if name in {"main_genre", "main_region"} else 0
        for name in ["main_genre", "main_region"]:
            if name in frame.columns:
                frame[name] = frame[name].astype("category")
        return frame

    def _ensure_feature_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        defaults: dict[str, Any] = {
            "popularity": 0,
            "age": 0,
            "rating": 0,
            "runtimeMinutes": 0,
            "num_translations": 0,
            "cluster": 0,
            "is_multiregional": 0,
            "director_name": None,
            "genres": "[]",
            "actors_list": "[]",
            "num_interactions": 0,
            "main_region": "UNKNOWN",
        }
        for name, value in defaults.items():
            if name not in result.columns:
                result[name] = value
        return result
