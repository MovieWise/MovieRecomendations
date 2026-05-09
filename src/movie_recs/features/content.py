"""Content feature builders."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def add_description_clusters(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    n_clusters: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Attach KMeans cluster assignments to a movie frame."""
    result = frame.copy()
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    result["cluster"] = model.fit_predict(embeddings)
    return result

