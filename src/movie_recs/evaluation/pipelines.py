"""Evaluation runners and report exporting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from movie_recs.metrics.ranking import Evaluator, EvaluationResult


def evaluate_model_predictions(
    truth_by_user: dict[int, set[int]],
    predictions: dict[int, list[int]],
    *,
    k_ndcg: int = 10,
    k_prec: int = 10,
    k_rec: int = 50,
) -> EvaluationResult:
    """Evaluate mapping-based recommendation outputs."""
    return Evaluator.evaluate(
        truth_by_user,
        predictions,
        k_ndcg=k_ndcg,
        k_prec=k_prec,
        k_rec=k_rec,
    )


def export_metrics_table(rows: Iterable[dict], path: str | Path) -> Path:
    """Save a comparison table as CSV."""
    frame = pd.DataFrame(list(rows))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output

