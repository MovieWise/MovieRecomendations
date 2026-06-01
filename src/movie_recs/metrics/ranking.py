"""Canonical ranking metrics used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def dcg(scores: Sequence[float]) -> float:
    """Compute discounted cumulative gain."""
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sum((2**values - 1) / np.log2(np.arange(values.size) + 2)))


def ndcg_at_k(gt_items: Iterable[int], predicted: Sequence[int], k: int = 10) -> float:
    """Compute NDCG for a single ranked list."""
    gt_set = set(gt_items)
    ranked = list(predicted[:k])
    relevance = np.array([1 if item in gt_set else 0 for item in ranked], dtype=float)
    rank_dcg = dcg(relevance)
    ideal_dcg = dcg(np.sort(relevance)[::-1])
    return 0.0 if ideal_dcg == 0 else rank_dcg / ideal_dcg


def precision_at_k(gt_items: Iterable[int], predicted: Sequence[int], k: int = 10) -> float:
    """Compute precision@k."""
    if k <= 0:
        return 0.0
    return len(set(predicted[:k]) & set(gt_items)) / float(k)


def recall_at_k(gt_items: Iterable[int], predicted: Sequence[int], k: int = 50) -> float:
    """Compute recall@k."""
    gt_set = set(gt_items)
    if not gt_set:
        return 0.0
    return len(set(predicted[:k]) & gt_set) / float(len(gt_set))


def evaluate_recommendations(
    truth_by_user: Mapping[int, Iterable[int]],
    recommendations: Mapping[int, Sequence[int]],
    k_ndcg: int = 10,
    k_prec: int = 10,
    k_rec: int = 50,
) -> tuple[float, float, float]:
    """Evaluate a recommendation dictionary keyed by user ID."""
    ndcg_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    for user_id, true_items in truth_by_user.items():
        if user_id not in recommendations:
            continue
        true_set = set(true_items)
        if not true_set:
            continue
        predicted = recommendations[user_id]
        ndcg_values.append(ndcg_at_k(true_set, predicted, k=k_ndcg))
        precision_values.append(precision_at_k(true_set, predicted, k=k_prec))
        recall_values.append(recall_at_k(true_set, predicted, k=k_rec))

    if not ndcg_values:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(ndcg_values)),
        float(np.mean(precision_values)),
        float(np.mean(recall_values)),
    )


def evaluate_grouped_frame(
    frame: pd.DataFrame,
    score_col: str,
    user_col: str = "userId",
    item_col: str = "movieId",
    label_col: str = "label",
    k_ndcg: int = 10,
    k_prec: int = 10,
    k_rec: int = 50,
) -> tuple[float, float, float]:
    """Evaluate ranking scores stored in a flat feature table."""
    truth_by_user: dict[int, set[int]] = {}
    recs_by_user: dict[int, list[int]] = {}

    for user_id, group in frame.groupby(user_col):
        truth = set(group[group[label_col] == 1][item_col])
        if not truth:
            continue
        truth_by_user[int(user_id)] = truth
        recs_by_user[int(user_id)] = group.sort_values(score_col, ascending=False)[item_col].tolist()

    return evaluate_recommendations(truth_by_user, recs_by_user, k_ndcg, k_prec, k_rec)


@dataclass(slots=True)
class EvaluationResult:
    """Structured ranking metrics."""

    ndcg: float
    precision: float
    recall: float


class Evaluator:
    """Adapter that evaluates either grouped frames or recommender dictionaries."""

    @staticmethod
    def evaluate(
        truth_or_frame: Mapping[int, Iterable[int]] | pd.DataFrame,
        predictions: Mapping[int, Sequence[int]] | None = None,
        *,
        score_col: str | None = None,
        k_ndcg: int = 10,
        k_prec: int = 10,
        k_rec: int = 50,
    ) -> EvaluationResult:
        """Evaluate ranking outputs and return a typed result."""
        if isinstance(truth_or_frame, pd.DataFrame):
            if score_col is None:
                raise ValueError("score_col is required when evaluating a DataFrame.")
            ndcg, precision, recall = evaluate_grouped_frame(
                truth_or_frame,
                score_col=score_col,
                k_ndcg=k_ndcg,
                k_prec=k_prec,
                k_rec=k_rec,
            )
        else:
            if predictions is None:
                raise ValueError("predictions are required when evaluating a mapping.")
            ndcg, precision, recall = evaluate_recommendations(
                truth_or_frame,
                predictions,
                k_ndcg=k_ndcg,
                k_prec=k_prec,
                k_rec=k_rec,
            )
        return EvaluationResult(ndcg=ndcg, precision=precision, recall=recall)
