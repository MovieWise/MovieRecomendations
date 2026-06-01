"""Error and robustness analysis helpers for ranking experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from movie_recs.metrics.ranking import evaluate_grouped_frame


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """Metric snapshot for one robustness scenario."""

    scenario: str
    ndcg: float
    precision: float
    recall: float
    top_k_jaccard: float


def add_rank_columns(
    frame: pd.DataFrame,
    score_col: str,
    baseline_col: str = "score",
    user_col: str = "userId",
) -> pd.DataFrame:
    """Attach model and baseline ranks within each user group."""
    result = frame.copy()
    result["model_rank"] = result.groupby(user_col)[score_col].rank(method="first", ascending=False).astype(int)
    result["baseline_rank"] = result.groupby(user_col)[baseline_col].rank(method="first", ascending=False).astype(int)
    return result


def categorize_error(row: pd.Series) -> str:
    """Assign a human-readable error category using available ranking features."""
    label = int(row.get("label", 0))
    popularity = float(row.get("popularity", 0.0) or 0.0)
    score = float(row.get("score", 0.0) or 0.0)
    num_interactions = float(row.get("num_interactions", 0.0) or 0.0)
    main_genre = str(row.get("main_genre", "Unknown"))
    main_region = str(row.get("main_region", "UNKNOWN"))
    has_director = int(row.get("has_director", 0) or 0)

    metadata_missing = (
        main_genre.lower() == "unknown"
        or main_region.upper() == "UNKNOWN"
        or has_director == 0
        or pd.isna(row.get("runtimeMinutes"))
        or pd.isna(row.get("rating"))
    )
    if metadata_missing:
        return "metadata missing"
    if num_interactions <= 5:
        return "weak user profile"
    if label == 0 and popularity >= 0.75:
        return "popularity bias"
    if label == 1 and popularity <= 0.25:
        return "missed niche positive"
    if label == 1 and score <= 0.05:
        return "low EASE score despite positive label"
    if main_genre.lower() == "unknown" or main_region.upper() == "UNKNOWN":
        return "genre/region mismatch"
    return "ranking boundary error"


def build_error_analysis(
    frame: pd.DataFrame,
    score_col: str,
    *,
    baseline_col: str = "score",
    user_col: str = "userId",
    item_col: str = "movieId",
    label_col: str = "label",
    top_k: int = 10,
    max_examples: int = 20,
) -> pd.DataFrame:
    """Return representative false positives and false negatives for top-k ranking."""
    ranked = add_rank_columns(frame, score_col=score_col, baseline_col=baseline_col, user_col=user_col)
    false_positives = ranked[(ranked["model_rank"] <= top_k) & (ranked[label_col] == 0)].copy()
    false_negatives = ranked[(ranked["model_rank"] > top_k) & (ranked[label_col] == 1)].copy()
    false_positives["error_type"] = "false_positive_top_k"
    false_negatives["error_type"] = "false_negative_outside_top_k"

    errors = pd.concat([false_positives, false_negatives], ignore_index=True)
    if errors.empty:
        return pd.DataFrame(
            columns=[
                user_col,
                item_col,
                "title",
                label_col,
                baseline_col,
                score_col,
                "baseline_rank",
                "model_rank",
                "error_type",
                "category",
                "explanation",
            ]
        )

    errors["category"] = errors.apply(categorize_error, axis=1)
    errors["rank_delta"] = errors["baseline_rank"] - errors["model_rank"]
    errors = errors.sort_values(["error_type", "model_rank", "rank_delta"], ascending=[True, True, False])
    errors = errors.head(max_examples).copy()
    errors["explanation"] = errors.apply(
        lambda row: (
            f"{row['error_type']} categorized as {row['category']}; "
            f"baseline_rank={row['baseline_rank']}, model_rank={row['model_rank']}."
        ),
        axis=1,
    )

    keep = [
        user_col,
        item_col,
        "title",
        label_col,
        baseline_col,
        score_col,
        "baseline_rank",
        "model_rank",
        "error_type",
        "category",
        "explanation",
    ]
    return errors[[col for col in keep if col in errors.columns]]


def build_error_report(errors: pd.DataFrame) -> str:
    """Build a compact markdown report for MLflow artifacts."""
    if errors.empty:
        return "# Error analysis\n\nNo top-k false positives or false negatives were found."
    counts = errors["category"].value_counts().to_dict()
    lines = ["# Error analysis", "", "## Category counts", ""]
    lines.extend(f"- {category}: {count}" for category, count in counts.items())
    lines.extend(["", "## Representative examples", ""])
    for row in errors.itertuples(index=False):
        movie_id = getattr(row, "movieId", "unknown")
        title = getattr(row, "title", "")
        error_type = getattr(row, "error_type", "")
        category = getattr(row, "category", "")
        explanation = getattr(row, "explanation", "")
        lines.append(f"- movieId={movie_id} {title}: {error_type}, {category}. {explanation}")
    return "\n".join(lines)


def mean_top_k_jaccard(
    baseline_frame: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    score_col: str,
    *,
    user_col: str = "userId",
    item_col: str = "movieId",
    top_k: int = 10,
) -> float:
    """Compute average top-k Jaccard overlap between two scored frames."""
    values: list[float] = []
    scenario_by_user = {user_id: group for user_id, group in scenario_frame.groupby(user_col)}
    for user_id, base_group in baseline_frame.groupby(user_col):
        scenario_group = scenario_by_user.get(user_id)
        if scenario_group is None:
            continue
        base_items = set(base_group.sort_values(score_col, ascending=False)[item_col].head(top_k))
        scenario_items = set(scenario_group.sort_values(score_col, ascending=False)[item_col].head(top_k))
        union = base_items | scenario_items
        values.append(1.0 if not union else len(base_items & scenario_items) / len(union))
    return float(np.mean(values)) if values else 0.0


def run_feature_robustness(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    predict_scores: Callable[[pd.DataFrame], Sequence[float]],
    *,
    score_col: str = "ranker_score",
    seed: int = 42,
    top_k: int = 10,
    k_ndcg: int = 10,
    k_prec: int = 15,
    k_rec: int = 50,
) -> pd.DataFrame:
    """Evaluate score stability under small feature perturbations."""
    rng = np.random.default_rng(seed)
    baseline = frame.copy()
    baseline[score_col] = np.asarray(predict_scores(baseline), dtype=float)
    base_ndcg, base_precision, base_recall = evaluate_grouped_frame(
        baseline,
        score_col=score_col,
        k_ndcg=k_ndcg,
        k_prec=k_prec,
        k_rec=k_rec,
    )
    results = [
        RobustnessResult("original", base_ndcg, base_precision, base_recall, 1.0),
    ]

    numeric_features = [
        col
        for col in feature_names
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if numeric_features:
        noisy = frame.copy()
        for col in numeric_features:
            scale = noisy[col].replace([np.inf, -np.inf], np.nan).std()
            scale = 1.0 if pd.isna(scale) or scale == 0 else float(scale)
            noisy[col] = noisy[col].astype(float) + rng.normal(0, 0.01 * scale, size=len(noisy))
        noisy[score_col] = np.asarray(predict_scores(noisy), dtype=float)
        ndcg, precision, recall = evaluate_grouped_frame(noisy, score_col=score_col, k_ndcg=k_ndcg, k_prec=k_prec, k_rec=k_rec)
        results.append(
            RobustnessResult(
                "numeric_noise_1pct",
                ndcg,
                precision,
                recall,
                mean_top_k_jaccard(baseline, noisy, score_col, top_k=top_k),
            )
        )

    degraded = frame.copy()
    for col in ["main_genre", "main_region"]:
        if col in degraded.columns:
            degraded[col] = pd.Series(["UNKNOWN"] * len(degraded), index=degraded.index).astype("category")
    if "has_director" in degraded.columns:
        degraded["has_director"] = 0
    degraded[score_col] = np.asarray(predict_scores(degraded), dtype=float)
    ndcg, precision, recall = evaluate_grouped_frame(degraded, score_col=score_col, k_ndcg=k_ndcg, k_prec=k_prec, k_rec=k_rec)
    results.append(
        RobustnessResult(
            "metadata_degraded",
            ndcg,
            precision,
            recall,
            mean_top_k_jaccard(baseline, degraded, score_col, top_k=top_k),
        )
    )

    return pd.DataFrame([asdict(result) for result in results])
