"""Split utilities reproduced from the notebooks."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def temporal_split(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    cutoff: int = 1514764800,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions by a UNIX timestamp cutoff."""
    train_df = frame.loc[frame[timestamp_col] < cutoff].copy()
    test_df = frame.loc[frame[timestamp_col] >= cutoff].copy()
    return train_df, test_df


def iterative_intersection_filter(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    max_iterations: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only user/item intersections in both splits until convergence."""
    train = train_df.copy()
    test = test_df.copy()

    for _ in range(max_iterations):
        prev_train_len = len(train)
        prev_test_len = len(test)

        common_users = np.intersect1d(train[user_col].unique(), test[user_col].unique())
        train = train[train[user_col].isin(common_users)].copy()
        test = test[test[user_col].isin(common_users)].copy()

        common_items = np.intersect1d(train[item_col].unique(), test[item_col].unique())
        train = train[train[item_col].isin(common_items)].copy()
        test = test[test[item_col].isin(common_items)].copy()

        if len(train) == prev_train_len and len(test) == prev_test_len:
            break

    return train, test


def group_split(
    frame: pd.DataFrame,
    group_col: str = "userId",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by group while keeping groups intact."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, groups=frame[group_col]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()

