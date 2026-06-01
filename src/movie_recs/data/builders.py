"""Feature and matrix builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


@dataclass(slots=True)
class EncodedInteractions:
    """Encoded interaction data and the corresponding encoders."""

    frame: pd.DataFrame
    user_encoder: LabelEncoder
    item_encoder: LabelEncoder


def encode_interactions(
    frame: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> EncodedInteractions:
    """Encode raw user and item identifiers."""
    result = frame.copy()
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    result["user_idx"] = user_encoder.fit_transform(result[user_col])
    result["item_idx"] = item_encoder.fit_transform(result[item_col])
    return EncodedInteractions(result, user_encoder, item_encoder)


def build_interaction_matrix(
    frame: pd.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
    value_col: str = "rating",
    n_users: int | None = None,
    n_items: int | None = None,
) -> csr_matrix:
    """Build a CSR user-item matrix."""
    if frame.empty:
        return csr_matrix((n_users or 0, n_items or 0))
    row_count = n_users if n_users is not None else frame[user_col].max() + 1
    # Keep one extra item slot by default for compatibility with unknown-item padding.
    col_count = n_items if n_items is not None else frame[item_col].max() + 2
    return csr_matrix(
        (frame[value_col], (frame[user_col], frame[item_col])),
        shape=(row_count, col_count),
    )
