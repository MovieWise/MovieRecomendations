"""Linear recommenders such as EASE and SLIM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse as sps
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet

from movie_recs.utils.io import load_pickle, save_pickle


class EASERecommender:
    """Embarrassingly Shallow Autoencoder recommender."""

    def __init__(self, reg_weight: float = 100.0) -> None:
        self.reg_weight = reg_weight
        self.weights: np.ndarray | None = None
        self.interactions: csr_matrix | None = None

    def fit(self, train_matrix: csr_matrix, *args, **kwargs) -> "EASERecommender":
        self.interactions = train_matrix.tocsr()
        gram = self.interactions.T @ self.interactions
        gram += self.reg_weight * sps.identity(gram.shape[0])
        precision = np.linalg.inv(gram.todense())
        weights = precision / (-np.diag(precision))
        np.fill_diagonal(weights, 0.0)
        self.weights = np.asarray(weights)
        return self

    def score_profile(self, encoded_items: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted.")
        vector = np.zeros(self.weights.shape[0], dtype=float)
        vector[encoded_items] = 1.0
        preds = np.asarray(vector @ self.weights).flatten()
        preds[encoded_items] = -np.inf
        return preds

    def recommend_encoded(self, user_index: int, top_k: int = 10) -> list[int]:
        if self.interactions is None or self.weights is None:
            raise RuntimeError("Model is not fitted.")
        user_row = self.interactions[user_index].toarray().flatten()
        scores = user_row.astype(np.float32) @ self.weights.astype(np.float32)
        seen = user_row.nonzero()[0]
        scores[seen] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        return self.recommend_encoded(user_id, top_k=top_k)

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        encoded = np.asarray(item_ids, dtype=int)
        scores = self.score_profile(encoded)
        return np.argsort(scores)[::-1][:top_k].tolist()

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "EASERecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model


class SLIMRecommender:
    """Sparse Linear Methods recommender."""

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.1, n_jobs: int = -1) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs
        self.weights: np.ndarray | None = None
        self.interactions: csr_matrix | None = None

    def fit(self, train_matrix: csr_matrix, *args, **kwargs) -> "SLIMRecommender":
        self.interactions = train_matrix.tocsr()
        n_items = self.interactions.shape[1]
        weights = np.zeros((n_items, n_items), dtype=np.float32)
        matrix_lil = self.interactions.tolil()

        def train_item(item_idx: int) -> tuple[int, np.ndarray]:
            target = self.interactions.getcol(item_idx).toarray().ravel()
            mask = target > 0
            if mask.sum() < 2:
                return item_idx, np.zeros(n_items, dtype=np.float32)
            features = matrix_lil.copy()
            features[:, item_idx] = 0
            features = features.tocsr()
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                positive=True,
                fit_intercept=False,
            )
            model.fit(features[mask], target[mask])
            coef = model.coef_
            coef[item_idx] = 0
            return item_idx, coef.astype(np.float32)

        results = Parallel(n_jobs=self.n_jobs)(delayed(train_item)(idx) for idx in range(n_items))
        for item_idx, coef in results:
            weights[item_idx] = coef
        self.weights = weights
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self.interactions is None or self.weights is None:
            raise RuntimeError("Model is not fitted.")
        scores = self.interactions[user_id].dot(self.weights).A1
        if exclude_seen:
            scores[self.interactions[user_id].indices] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        if self.weights is None:
            raise RuntimeError("Model is not fitted.")
        vector = np.zeros(self.weights.shape[0], dtype=float)
        vector[item_ids] = 1.0
        scores = vector.dot(self.weights)
        scores[item_ids] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "SLIMRecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model

