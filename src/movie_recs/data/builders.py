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
) -> csr_matrix:
    """Build a CSR user-item matrix."""
    return csr_matrix(
        (frame[value_col], (frame[user_col], frame[item_col])),
        shape=(frame[user_col].max() + 1, frame[item_col].max() + 1),
    )

