"""MLflow-aware training pipeline for the final EASE + LightGBM ranker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from movie_recs.config.schemas import ExperimentConfig
from movie_recs.data.builders import build_interaction_matrix, encode_interactions
from movie_recs.evaluation.error_analysis import build_error_analysis, build_error_report, run_feature_robustness
from movie_recs.metrics.ranking import evaluate_grouped_frame
from movie_recs.models.hybrid_rankers import LightGBMHybridRanker, get_ease_candidates
from movie_recs.models.linear import EASERecommender
from movie_recs.preprocessing.ranking_dataset import RankingDatasetBuilder
from movie_recs.utils.io import ensure_dir, save_joblib, save_json


PRD_FEATURES = [
    "score",
    "popularity",
    "age",
    "rating",
    "runtimeMinutes",
    "num_translations",
    "cluster",
    "has_director",
    "num_interactions",
    "main_genre",
    "main_region",
    "main_actor_popularity",
    "user_main_genre_count",
    "user_main_genre_share",
    "user_main_region_count",
    "user_main_region_share",
    "user_cluster_count",
    "user_cluster_share",
]

USER_COUNTER_FEATURES = [
    "user_main_genre_count",
    "user_main_genre_share",
    "user_main_region_count",
    "user_main_region_share",
    "user_cluster_count",
    "user_cluster_share",
]


@dataclass(slots=True)
class HybridTrainingResult:
    """Final training result and artifact locations."""

    run_name: str
    artifact_dir: Path
    metrics: dict[str, float | None]
    artifacts: dict[str, Path]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _project_root() / value


def _read_table(path: str | Path) -> pd.DataFrame:
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required dataset is missing: {resolved}")
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)


def _safe_auc(frame: pd.DataFrame, score_col: str, label_col: str = "label") -> float | None:
    labels = frame[label_col]
    if labels.nunique(dropna=True) < 2:
        return None
    return float(roc_auc_score(labels, frame[score_col]))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _flatten_mapping(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, name))
        else:
            flat[name] = value
    return flat


def _log_metric_safe(mlflow: Any, name: str, value: float | None) -> None:
    if value is None:
        return
    numeric = float(value)
    if np.isfinite(numeric):
        mlflow.log_metric(name, numeric)


def _prepare_movies_features(ratings: pd.DataFrame, movies: pd.DataFrame, content: pd.DataFrame) -> pd.DataFrame:
    result = movies.copy()
    if "genres" in result.columns:
        result["genres"] = result["genres"].apply(lambda value: str(value).split("|") if "|" in str(value) else value)
    counts = ratings.groupby("movieId").size().rename("popularity_count")
    total_count = max(float(counts.sum()), 1.0)
    result = result.merge(counts.reset_index(), on="movieId", how="left")
    result["popularity"] = result["popularity_count"].fillna(0) / total_count
    result = result.drop(columns=["popularity_count"])
    if "rating" not in result.columns:
        result = result.merge(ratings.groupby("movieId")["rating"].mean().reset_index(), on="movieId", how="left")
    if "age" not in result.columns and "title" in result.columns:
        years = result["title"].astype(str).str.extract(r"\((\d{4})\)")[0].astype(float)
        result["age"] = datetime.now(timezone.utc).year - years
    if not content.empty:
        content_dedup = content.drop_duplicates("movieId") if "movieId" in content.columns else content
        result = result.merge(content_dedup, on="movieId", how="left", suffixes=("", "_content"))
        for col in result.columns:
            if col.endswith("_content"):
                base_col = col.removesuffix("_content")
                if base_col in result.columns:
                    result[base_col] = result[base_col].combine_first(result[col])
                    result = result.drop(columns=[col])
    return result


def _split_ratings_temporally(ratings: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in ratings.columns:
        raise ValueError("ratings.csv must contain a timestamp column for temporal split.")
    cutoff_ts = pd.Timestamp(cutoff_date).timestamp()
    timestamps = pd.to_numeric(ratings["timestamp"], errors="coerce")
    train = ratings[timestamps < cutoff_ts].copy()
    test = ratings[timestamps >= cutoff_ts].copy()
    included = np.intersect1d(train["userId"].unique(), test["userId"].unique())
    return train[train["userId"].isin(included)].copy(), test[test["userId"].isin(included)].copy()


def _user_counter_params(config: ExperimentConfig) -> dict[str, Any]:
    return dict(config.training_params.get("user_counter_features", {}))


def _user_counter_features_enabled(config: ExperimentConfig) -> bool:
    return bool(_user_counter_params(config).get("enabled", False))


def _features_required_before_user_counters(config: ExperimentConfig) -> list[str]:
    return [feature for feature in _feature_names(config) if feature not in USER_COUNTER_FEATURES]


def _build_item_metadata_frame(*frames: pd.DataFrame) -> pd.DataFrame:
    columns = ["movieId", "main_genre", "main_region", "cluster"]
    available_frames = [frame[[col for col in columns if col in frame.columns]].copy() for frame in frames if frame is not None and not frame.empty]
    if not available_frames:
        return pd.DataFrame(columns=columns)
    metadata = pd.concat(available_frames, ignore_index=True)
    for col in columns:
        if col not in metadata.columns:
            metadata[col] = np.nan
    metadata["main_genre"] = metadata["main_genre"].astype("object").where(metadata["main_genre"].notna(), "UNKNOWN")
    metadata["main_region"] = metadata["main_region"].astype("object").where(metadata["main_region"].notna(), "UNKNOWN")
    metadata["cluster"] = pd.to_numeric(metadata["cluster"], errors="coerce").fillna(-1).astype(float)
    return metadata.dropna(subset=["movieId"]).drop_duplicates("movieId")[columns]


def _filter_relevant_history(
    ratings: pd.DataFrame,
    *,
    cutoff_date: str,
    min_rating: float,
    user_ids: set[int] | None = None,
) -> pd.DataFrame:
    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings history is missing required columns: {sorted(missing)}")
    result = ratings.copy()
    if user_ids is not None:
        result = result[result["userId"].isin(user_ids)]
    cutoff_ts = pd.Timestamp(cutoff_date).timestamp()
    timestamps = pd.to_numeric(result["timestamp"], errors="coerce")
    ratings_numeric = pd.to_numeric(result["rating"], errors="coerce")
    result = result[(timestamps < cutoff_ts) & (ratings_numeric >= min_rating)].copy()
    return result[["userId", "movieId", "rating", "timestamp"]]


def _add_dimension_counter_features(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    dimension_col: str,
    prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    count_col = f"{prefix}_count"
    share_col = f"{prefix}_share"
    total_col = f"{prefix}_total"
    if history.empty or dimension_col not in history.columns:
        result[count_col] = 0.0
        result[share_col] = 0.0
        return result

    history_dim = history[["userId", dimension_col]].copy()
    frame_dim = result[["userId", dimension_col]].copy()
    if dimension_col in {"main_genre", "main_region"}:
        history_dim[dimension_col] = history_dim[dimension_col].astype("object").where(history_dim[dimension_col].notna(), "UNKNOWN")
        frame_dim[dimension_col] = frame_dim[dimension_col].astype("object").where(frame_dim[dimension_col].notna(), "UNKNOWN")
    else:
        history_dim[dimension_col] = pd.to_numeric(history_dim[dimension_col], errors="coerce").fillna(-1).astype(float)
        frame_dim[dimension_col] = pd.to_numeric(frame_dim[dimension_col], errors="coerce").fillna(-1).astype(float)
    result[dimension_col] = frame_dim[dimension_col]

    counts = history_dim.groupby(["userId", dimension_col]).size().reset_index(name=count_col)
    totals = history_dim.groupby("userId").size().reset_index(name=total_col)
    result = result.merge(counts, on=["userId", dimension_col], how="left")
    result = result.merge(totals, on="userId", how="left")
    result[count_col] = result[count_col].fillna(0).astype(float)
    result[total_col] = result[total_col].fillna(0).astype(float)
    result[share_col] = np.divide(
        result[count_col],
        result[total_col],
        out=np.zeros(len(result), dtype=float),
        where=result[total_col].to_numpy() > 0,
    )
    return result.drop(columns=[total_col])


def add_user_counter_features(
    frame: pd.DataFrame,
    ratings: pd.DataFrame,
    item_metadata: pd.DataFrame,
    *,
    cutoff_date: str = "2018-01-01",
    min_rating: float = 3.0,
) -> pd.DataFrame:
    """Add leakage-free user profile counters from historical ratings."""
    if frame.empty:
        result = frame.copy()
        for feature in USER_COUNTER_FEATURES:
            result[feature] = 0.0
        return result
    users = {int(user_id) for user_id in frame["userId"].dropna().unique()}
    history = _filter_relevant_history(ratings, cutoff_date=cutoff_date, min_rating=min_rating, user_ids=users)
    metadata = item_metadata[["movieId", "main_genre", "main_region", "cluster"]].copy()
    history = history.merge(metadata, on="movieId", how="inner")
    result = _add_dimension_counter_features(frame, history, "main_genre", "user_main_genre")
    result = _add_dimension_counter_features(result, history, "main_region", "user_main_region")
    result = _add_dimension_counter_features(result, history, "cluster", "user_cluster")
    for feature in USER_COUNTER_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(0.0)
    return result


def _apply_user_counter_features(
    config: ExperimentConfig,
    frame: pd.DataFrame,
    external_test: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    params = _user_counter_params(config)
    ratings_path = _resolve_path(config.data.ratings)
    if not ratings_path.exists():
        raise FileNotFoundError(
            "User counter features are enabled, but ratings history is missing: "
            f"{ratings_path}. Put ratings.csv there or set training_params.user_counter_features.enabled=false."
        )
    ratings = _read_table(ratings_path)
    cutoff_date = str(params.get("cutoff_date", config.split_params.get("cutoff_date", "2018-01-01")))
    min_rating = float(params.get("min_rating", 3.0))
    item_metadata = _build_item_metadata_frame(frame, external_test if external_test is not None else pd.DataFrame())
    enriched_frame = add_user_counter_features(
        frame,
        ratings,
        item_metadata,
        cutoff_date=cutoff_date,
        min_rating=min_rating,
    )
    enriched_test = None
    if external_test is not None:
        enriched_test = add_user_counter_features(
            external_test,
            ratings,
            item_metadata,
            cutoff_date=cutoff_date,
            min_rating=min_rating,
        )
    metadata = {
        "user_counter_features_enabled": True,
        "user_counter_min_rating": min_rating,
        "user_counter_cutoff_date": cutoff_date,
        "user_counter_ratings_path": str(ratings_path),
        "user_counter_metadata_items": int(item_metadata["movieId"].nunique()) if "movieId" in item_metadata.columns else 0,
    }
    return enriched_frame, enriched_test, metadata


def build_hybrid_training_frame(config: ExperimentConfig, limit_users: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a reranking table from local raw and processed data."""
    ratings = _read_table(config.data.ratings)
    movies = _read_table(config.data.movies)
    content_path = _resolve_path(config.data.content)
    content = _read_table(content_path) if content_path.exists() else pd.DataFrame()

    cutoff_date = str(config.split_params.get("cutoff_date", "2018-01-01"))
    train_ratings, test_ratings = _split_ratings_temporally(ratings, cutoff_date)
    if limit_users is not None:
        selected_users = sorted(train_ratings["userId"].unique().tolist())[:limit_users]
        train_ratings = train_ratings[train_ratings["userId"].isin(selected_users)].copy()
        test_ratings = test_ratings[test_ratings["userId"].isin(selected_users)].copy()

    train_grouped = (
        train_ratings.groupby("userId")
        .apply(lambda frame: list(zip(frame["movieId"], frame["rating"])))
        .reset_index(name="train_interactions")
    )
    test_grouped = (
        test_ratings.groupby("userId")
        .apply(lambda frame: list(zip(frame["movieId"], frame["rating"])))
        .reset_index(name="test_interactions")
    )
    joined = train_grouped.merge(test_grouped, on="userId", how="inner")

    top_items = int(config.training_params.get("ease_top_items", 25000))
    popular_items = train_ratings["movieId"].value_counts().head(top_items).index
    train_pop = train_ratings[train_ratings["movieId"].isin(popular_items)].copy()
    encoded = encode_interactions(train_pop)
    matrix = build_interaction_matrix(
        encoded.frame,
        value_col="rating",
        n_users=len(encoded.user_encoder.classes_),
        n_items=len(encoded.item_encoder.classes_),
    )

    ease = EASERecommender(reg_weight=float(config.model_params.get("reg_weight", 100.0))).fit(matrix)
    candidate_top_k = int(config.training_params.get("candidate_top_k", 100))
    joined["candidates_with_scores"] = joined["train_interactions"].apply(
        lambda items: get_ease_candidates(items, encoded.item_encoder, ease.weights, top_k=candidate_top_k)
    )
    exploded = joined[["userId", "candidates_with_scores"]].explode("candidates_with_scores").dropna()
    candidate_cols = pd.DataFrame(
        exploded["candidates_with_scores"].tolist(),
        index=exploded.index,
        columns=["movieId", "score"],
    )
    candidates = pd.concat([exploded["userId"], candidate_cols], axis=1)
    movie_features = _prepare_movies_features(train_ratings, movies, content)
    rerank_frame = candidates.merge(movie_features, on="movieId", how="left")

    truth_items = {
        int(row.userId): {int(item_id) for item_id, rating in row.test_interactions if float(rating) >= 3.0}
        for row in joined.itertuples()
    }
    num_interactions = {int(row.userId): len(row.train_interactions) for row in joined.itertuples()}
    rerank_frame["num_interactions"] = rerank_frame["userId"].map(num_interactions)
    builder = RankingDatasetBuilder(feature_names=tuple(_feature_names(config)))
    rerank_frame = builder.build(rerank_frame, truth_items=truth_items)

    metadata = {
        "ease_model": ease,
        "item_encoder": encoded.item_encoder,
        "user_encoder": encoded.user_encoder,
        "ratings_rows": len(ratings),
        "train_interactions": len(train_ratings),
        "test_interactions": len(test_ratings),
        "candidate_rows": len(rerank_frame),
    }
    return rerank_frame, metadata


def _load_or_build_frame(config: ExperimentConfig, limit_users: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    hybrid_path = _resolve_path(config.data.hybrid_table)
    if hybrid_path.exists():
        frame = _read_table(hybrid_path)
        if limit_users is not None and "userId" in frame.columns:
            selected_users = sorted(frame["userId"].unique().tolist())[:limit_users]
            frame = frame[frame["userId"].isin(selected_users)].copy()
        metadata = {"hybrid_table": str(hybrid_path), "candidate_rows": len(frame)}
        if "label" not in frame.columns:
            raise ValueError(f"{hybrid_path} exists but does not contain label column.")
        if all(col in frame.columns for col in _features_required_before_user_counters(config)):
            return frame, metadata
        return RankingDatasetBuilder(feature_names=tuple(_feature_names(config))).build(frame), metadata
    return build_hybrid_training_frame(config, limit_users=limit_users)


def _prepare_cached_hybrid_frame(
    path: str | Path,
    config: ExperimentConfig,
    limit_users: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = _resolve_path(path)
    frame = _read_table(resolved)
    if limit_users is not None and "userId" in frame.columns:
        selected_users = sorted(frame["userId"].unique().tolist())[:limit_users]
        frame = frame[frame["userId"].isin(selected_users)].copy()
    if "label" not in frame.columns:
        raise ValueError(f"{resolved} exists but does not contain label column.")
    if not all(col in frame.columns for col in _features_required_before_user_counters(config)):
        frame = RankingDatasetBuilder(feature_names=tuple(_feature_names(config))).build(frame)
    metadata = {
        "hybrid_table": str(resolved),
        "candidate_rows": len(frame),
        "users": int(frame["userId"].nunique()) if "userId" in frame.columns else 0,
    }
    return frame, metadata


def _load_train_and_test_frames(
    config: ExperimentConfig,
    limit_users: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    train_path = _resolve_path(config.data.hybrid_table)
    test_path = _resolve_path(config.data.hybrid_test_table)
    if train_path.exists() and test_path.exists():
        train_frame, train_meta = _prepare_cached_hybrid_frame(train_path, config, limit_users)
        test_frame, test_meta = _prepare_cached_hybrid_frame(test_path, config, limit_users)
        metadata = {
            "train_hybrid_table": train_meta["hybrid_table"],
            "test_hybrid_table": test_meta["hybrid_table"],
            "train_candidate_rows": train_meta["candidate_rows"],
            "test_candidate_rows": test_meta["candidate_rows"],
            "train_users": train_meta["users"],
            "test_users": test_meta["users"],
        }
        return train_frame, test_frame, metadata
    frame, metadata = _load_or_build_frame(config, limit_users=limit_users)
    metadata["test_hybrid_table"] = ""
    return frame, None, metadata


def _feature_names(config: ExperimentConfig) -> list[str]:
    configured = config.training_params.get("features")
    features = list(configured or PRD_FEATURES)
    if not _user_counter_features_enabled(config):
        features = [feature for feature in features if feature not in USER_COUNTER_FEATURES]
    return features


def _split_train_validation_frame(config: ExperimentConfig, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config.seed)
    validation_size = float(config.training_params.get("validation_size", 0.2))
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, groups=frame["userId"]))
    return frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()


def _split_ranker_frame(config: ExperimentConfig, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config.seed)
    test_size = float(config.split_params.get("test_size", 0.25))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(frame, groups=frame["userId"]))
    train_val = frame.iloc[train_val_idx].copy()
    test = frame.iloc[test_idx].copy()
    train, val = _split_train_validation_frame(config, train_val)
    return train, val, test


def _prepare_ranker_frame(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in feature_names:
        if col not in result.columns:
            result[col] = "UNKNOWN" if col in {"main_genre", "main_region"} else 0
    for col in ["main_genre", "main_region"]:
        if col in result.columns:
            result[col] = result[col].astype("object").where(result[col].notna(), "UNKNOWN").astype("category")
    for col in feature_names:
        if col not in {"main_genre", "main_region"}:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result


def _fit_lgbm_ranker(config: ExperimentConfig, train: pd.DataFrame, val: pd.DataFrame, feature_names: Sequence[str]) -> LightGBMHybridRanker:
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise RuntimeError("Install lightgbm to train EASE + LightGBM Ranker.") from exc

    ranker_params = dict(config.training_params.get("ranker_params", {}))
    ranker_params.setdefault("objective", "lambdarank")
    ranker_params.setdefault("n_estimators", 500)
    ranker_params.setdefault("max_depth", 3)
    ranker_params.setdefault("learning_rate", 0.1)
    ranker_params.setdefault("random_state", config.seed)
    ranker_params.setdefault("n_jobs", -1)
    ranker_params.setdefault("verbose", -1)

    train_sorted = train.sort_values("userId").reset_index(drop=True)
    val_sorted = val.sort_values("userId").reset_index(drop=True)
    train_groups = train_sorted.groupby("userId").size().values
    val_groups = val_sorted.groupby("userId").size().values
    model = LGBMRanker(**ranker_params)
    model.fit(
        train_sorted[list(feature_names)],
        train_sorted["label"],
        group=train_groups,
        eval_set=[(val_sorted[list(feature_names)], val_sorted["label"])],
        eval_group=[val_groups],
        callbacks=[lgb.log_evaluation(50)],
    )
    return LightGBMHybridRanker(feature_names=list(feature_names), model=model)


def _score_frame(ranker: LightGBMHybridRanker, frame: pd.DataFrame, score_col: str = "ranker_score") -> pd.DataFrame:
    result = frame.copy()
    result[score_col] = ranker.predict_scores(result)
    return result


def _collect_metrics(
    frames: Mapping[str, pd.DataFrame],
    *,
    ranker_score_col: str = "ranker_score",
    baseline_score_col: str = "score",
    k_ndcg: int = 10,
    k_prec: int = 15,
    k_rec: int = 50,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for split, frame in frames.items():
        ndcg, precision, recall = evaluate_grouped_frame(frame, ranker_score_col, k_ndcg=k_ndcg, k_prec=k_prec, k_rec=k_rec)
        base_ndcg, base_precision, base_recall = evaluate_grouped_frame(frame, baseline_score_col, k_ndcg=k_ndcg, k_prec=k_prec, k_rec=k_rec)
        metrics[f"{split}.ndcg@{k_ndcg}"] = ndcg
        metrics[f"{split}.precision@{k_prec}"] = precision
        metrics[f"{split}.recall@{k_rec}"] = recall
        metrics[f"{split}.auc"] = _safe_auc(frame, ranker_score_col)
        metrics[f"{split}.ease_baseline.ndcg@{k_ndcg}"] = base_ndcg
        metrics[f"{split}.ease_baseline.precision@{k_prec}"] = base_precision
        metrics[f"{split}.ease_baseline.recall@{k_rec}"] = base_recall
        metrics[f"{split}.delta_ndcg@{k_ndcg}"] = ndcg - base_ndcg
    return metrics


def _save_plot_artifacts(test: pd.DataFrame, ranker: LightGBMHybridRanker, artifact_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt
        from lightgbm import plot_importance
    except Exception:
        return paths

    plots_dir = ensure_dir(artifact_dir / "plots")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(test["score"], bins=40, alpha=0.75)
    axes[0].set_title("EASE score distribution")
    axes[1].hist(test["ranker_score"], bins=40, alpha=0.75)
    axes[1].set_title("LGBM ranker score distribution")
    score_path = plots_dir / "score_distributions.png"
    fig.tight_layout()
    fig.savefig(score_path)
    plt.close(fig)
    paths["score_distributions"] = score_path

    if ranker.model is not None and hasattr(ranker.model, "booster_"):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_importance(ranker.model, ax=ax, title="LGBM Ranker Feature Importance")
        importance_path = plots_dir / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(importance_path)
        plt.close(fig)
        paths["feature_importance"] = importance_path
    return paths


def _save_artifacts(
    config: ExperimentConfig,
    ranker: LightGBMHybridRanker,
    frames: Mapping[str, pd.DataFrame],
    metrics: dict[str, float | None],
    metadata: dict[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = ensure_dir(Path(config.artifacts.root) / run_id)
    paths: dict[str, Path] = {}

    paths["ranker_pickle"] = ranker.save(artifact_dir / "model" / "LightGBMHybridRanker.pkl")
    ease_model = metadata.get("ease_model")
    if ease_model is not None and getattr(ease_model, "weights", None) is not None:
        ease_path = artifact_dir / "model" / "ease_weights_f16.npy"
        ensure_dir(ease_path.parent)
        np.save(ease_path, ease_model.weights.astype(np.float16))
        paths["ease_weights"] = ease_path
    for encoder_name in ["item_encoder", "user_encoder"]:
        if encoder_name in metadata:
            paths[encoder_name] = save_joblib(metadata[encoder_name], artifact_dir / "encoders" / f"{encoder_name}.joblib")

    paths["feature_schema"] = save_json({"features": list(feature_names)}, artifact_dir / "features" / "feature_schema.json")
    paths["config_snapshot"] = save_json(_to_jsonable(config.to_dict()), artifact_dir / "config_snapshot.json")
    paths["metrics_json"] = save_json(metrics, artifact_dir / "metrics" / "metrics.json")

    prediction_examples = frames["test"].sort_values(["userId", "ranker_score"], ascending=[True, False]).groupby("userId").head(5)
    examples_path = artifact_dir / "predictions" / "prediction_examples.csv"
    ensure_dir(examples_path.parent)
    prediction_examples.head(500).to_csv(examples_path, index=False)
    paths["prediction_examples"] = examples_path

    errors = build_error_analysis(frames["test"], "ranker_score", max_examples=20)
    errors_path = artifact_dir / "analysis" / "error_analysis.csv"
    ensure_dir(errors_path.parent)
    errors.to_csv(errors_path, index=False)
    paths["error_analysis"] = errors_path
    report_path = artifact_dir / "analysis" / "error_analysis.md"
    report_path.write_text(build_error_report(errors), encoding="utf-8")
    paths["error_report"] = report_path

    robustness = run_feature_robustness(
        frames["test"],
        feature_names,
        ranker.predict_scores,
        k_ndcg=int(config.evaluation_params.get("k_ndcg", 10)),
        k_prec=int(config.evaluation_params.get("k_prec", 15)),
        k_rec=int(config.evaluation_params.get("k_rec", 50)),
    )
    robustness_path = artifact_dir / "analysis" / "robustness_report.csv"
    robustness.to_csv(robustness_path, index=False)
    paths["robustness_report"] = robustness_path

    paths.update(_save_plot_artifacts(frames["test"], ranker, artifact_dir))
    return paths


def _log_to_mlflow(
    config: ExperimentConfig,
    metrics: dict[str, float | None],
    artifacts: Mapping[str, Path],
    metadata: Mapping[str, Any],
    ranker: LightGBMHybridRanker,
    run_name: str,
) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError("Install mlflow and boto3 to log the PRD experiment.") from exc

    if config.tracking.tracking_uri:
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", config.tracking.experiment_name))
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(
            {
                "model_name": "ease_lgb_ranker",
                "stage": config.tracking.prd_tag,
                "prd": "true",
                "final_model": "true",
            }
        )
        flat_config = _flatten_mapping(_to_jsonable(config.to_dict()))
        for name, value in flat_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(name[:250], value)
        for name, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"data.{name}"[:250], value)
        for name, value in metrics.items():
            _log_metric_safe(mlflow, name, value)
        for path in artifacts.values():
            mlflow.log_artifact(str(path))
        if ranker.model is not None:
            model_info = mlflow.sklearn.log_model(
                ranker.model,
                artifact_path="sklearn_lgbm_ranker",
                registered_model_name=config.tracking.registered_model_name,
            )
            client = mlflow.tracking.MlflowClient()
            client.set_tag(active_run.info.run_id, "mlflow_model_uri", model_info.model_uri)
            try:
                versions = client.search_model_versions(f"name = '{config.tracking.registered_model_name}'")
                latest = sorted(versions, key=lambda version: int(version.version))[-1]
                client.set_model_version_tag(config.tracking.registered_model_name, latest.version, "stage", config.tracking.prd_tag)
                client.set_model_version_tag(config.tracking.registered_model_name, latest.version, "final_model", "true")
            except Exception:
                pass


def train_prd_hybrid_ranker(
    config: ExperimentConfig,
    *,
    log_to_mlflow: bool = False,
    limit_users: int | None = None,
) -> HybridTrainingResult:
    """Train, evaluate and optionally log the PRD EASE + LGBM ranker."""
    config = config.with_default_artifacts()
    feature_names = _feature_names(config)
    frame, external_test, metadata = _load_train_and_test_frames(config, limit_users=limit_users)
    if _user_counter_features_enabled(config):
        frame, external_test, counter_metadata = _apply_user_counter_features(config, frame, external_test)
        metadata.update(counter_metadata)
    if external_test is None:
        train, val, test = _split_ranker_frame(config, frame)
    else:
        train, val = _split_train_validation_frame(config, frame)
        test = external_test
    train = _prepare_ranker_frame(train, feature_names)
    val = _prepare_ranker_frame(val, feature_names)
    test = _prepare_ranker_frame(test, feature_names)

    ranker = _fit_lgbm_ranker(config, train, val, feature_names)
    scored_frames = {
        "train": _score_frame(ranker, train),
        "validation": _score_frame(ranker, val),
        "test": _score_frame(ranker, test),
    }
    metrics = _collect_metrics(
        scored_frames,
        k_ndcg=int(config.evaluation_params.get("k_ndcg", 10)),
        k_prec=int(config.evaluation_params.get("k_prec", 15)),
        k_rec=int(config.evaluation_params.get("k_rec", 50)),
    )
    artifacts = _save_artifacts(config, ranker, scored_frames, metrics, metadata, feature_names)
    artifact_dir = artifacts["ranker_pickle"].parents[1]
    run_name = f"{config.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if log_to_mlflow:
        _log_to_mlflow(config, metrics, artifacts, metadata, ranker, run_name)
    return HybridTrainingResult(run_name=run_name, artifact_dir=artifact_dir, metrics=metrics, artifacts=dict(artifacts))


def load_prd_model_and_predict(
    *,
    experiment_name: str = "movie-recs-prd",
    registered_model_name: str = "MovieRecs_EASE_LGB_Ranker",
    prd_tag: str = "PRD",
    sample: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load the latest MLflow model version tagged PRD and score a sample frame."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install mlflow to load a PRD model.") from exc

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name = '{registered_model_name}'")
    prd_versions = [version for version in versions if version.tags.get("stage") == prd_tag]
    if not prd_versions:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"Experiment not found: {experiment_name}")
        runs = client.search_runs([experiment.experiment_id], filter_string=f"tags.stage = '{prd_tag}'", max_results=1)
        if not runs:
            raise RuntimeError(f"No PRD run found in experiment {experiment_name}.")
        model_uri = runs[0].data.tags.get("mlflow_model_uri")
        if model_uri is None:
            raise RuntimeError("PRD run does not contain mlflow_model_uri tag.")
    else:
        latest = sorted(prd_versions, key=lambda version: int(version.version))[-1]
        model_uri = f"models:/{registered_model_name}/{latest.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    if sample is None:
        sample = pd.DataFrame([{feature: 0 for feature in PRD_FEATURES}])
        sample["main_genre"] = "UNKNOWN"
        sample["main_region"] = "UNKNOWN"
    result = sample.copy()
    result["prediction"] = model.predict(sample)
    return result
