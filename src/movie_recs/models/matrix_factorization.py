"""Matrix factorization recommenders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from movie_recs.utils.io import load_pickle, save_pickle


class PureSVDRecommender:
    """PureSVD recommender from the base model notebooks."""

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors
        self.U: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.Vt: np.ndarray | None = None
        self.train_matrix: csr_matrix | None = None

    def fit(self, train_matrix: csr_matrix, *args, **kwargs) -> "PureSVDRecommender":
        self.train_matrix = train_matrix.tocsr()
        U, S, Vt = svds(self.train_matrix, k=self.n_factors)
        self.U = U
        self.S = np.diag(S)
        self.Vt = Vt
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self.U is None or self.S is None or self.Vt is None or self.train_matrix is None:
            raise RuntimeError("Model is not fitted.")
        scores = (self.U[user_id] @ (self.S @ self.Vt)).copy()
        if exclude_seen:
            scores[self.train_matrix[user_id].nonzero()[1]] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        if self.Vt is None:
            raise RuntimeError("Model is not fitted.")
        n_items = self.Vt.shape[1]
        user_vector = np.zeros(n_items, dtype=float)
        user_vector[item_ids] = 1.0
        V = self.Vt.T
        scores = user_vector @ V @ V.T
        scores[item_ids] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "PureSVDRecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model
