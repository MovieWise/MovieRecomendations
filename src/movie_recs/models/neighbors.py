"""User-based and item-based kNN recommenders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

from movie_recs.utils.io import load_pickle, save_pickle


class ItemKNNRecommender:
    """Fast item-kNN recommender from the notebook implementation."""

    def __init__(self, top_neighbors: int = 10) -> None:
        self.top_neighbors = top_neighbors
        self.train_matrix: csr_matrix | None = None
        self.item_similarity: csr_matrix | None = None

    def fit(self, train_matrix: csr_matrix, *args, **kwargs) -> "ItemKNNRecommender":
        self.train_matrix = train_matrix.tocsr()
        similarity = cosine_similarity(self.train_matrix.T, dense_output=False)
        top = lil_matrix(similarity.shape)
        for item_idx in range(similarity.shape[0]):
            row = similarity[item_idx].toarray().flatten()
            row[item_idx] = -np.inf
            top_indices = np.argsort(row)[::-1][: self.top_neighbors]
            top[item_idx, top_indices] = similarity[item_idx, top_indices]
        self.item_similarity = top.tocsr()
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self.train_matrix is None or self.item_similarity is None:
            raise RuntimeError("Model is not fitted.")
        scores = self.train_matrix[user_id].dot(self.item_similarity).A1
        if exclude_seen:
            scores[self.train_matrix[user_id].indices] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        if self.item_similarity is None:
            raise RuntimeError("Model is not fitted.")
        vector = np.zeros(self.item_similarity.shape[0], dtype=float)
        vector[item_ids] = 1.0
        scores = vector @ self.item_similarity.toarray()
        scores[item_ids] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "ItemKNNRecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model


class UserKNNRecommender:
    """Fast user-kNN recommender from the notebook implementation."""

    def __init__(self, top_neighbors: int = 10) -> None:
        self.top_neighbors = top_neighbors
        self.train_matrix: csr_matrix | None = None
        self.user_similarity: csr_matrix | None = None

    def fit(self, train_matrix: csr_matrix, *args, **kwargs) -> "UserKNNRecommender":
        self.train_matrix = train_matrix.tocsr()
        similarity = cosine_similarity(self.train_matrix, dense_output=False)
        top = lil_matrix(similarity.shape)
        for user_idx in range(similarity.shape[0]):
            row = similarity[user_idx].toarray().flatten()
            row[user_idx] = -np.inf
            top_indices = np.argsort(row)[::-1][: self.top_neighbors]
            top[user_idx, top_indices] = similarity[user_idx, top_indices]
        self.user_similarity = top.tocsr()
        return self

    def recommend(self, user_id: int, top_k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self.train_matrix is None or self.user_similarity is None:
            raise RuntimeError("Model is not fitted.")
        scores = self.user_similarity[user_id].dot(self.train_matrix).A1
        if exclude_seen:
            scores[self.train_matrix[user_id].indices] = -np.inf
        return np.argsort(scores)[::-1][:top_k].tolist()

    def recommend_for_profile(self, item_ids: list[int], top_k: int = 10) -> list[int]:
        raise NotImplementedError("Profile-based inference is not supported for UserKNNRecommender.")

    def save(self, path: str | Path) -> Path:
        return save_pickle(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "UserKNNRecommender":
        model = cls()
        model.__dict__.update(load_pickle(path))
        return model

