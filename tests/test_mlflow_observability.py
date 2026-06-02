from __future__ import annotations

import os

import pandas as pd
import pytest

from movie_recs.config.loader import load_experiment_config
from movie_recs.evaluation.error_analysis import build_error_analysis, run_feature_robustness
from movie_recs.training.hybrid_mlflow import (
    PRD_FEATURES,
    USER_COUNTER_FEATURES,
    _apply_user_counter_features,
    add_user_counter_features,
)


def test_hybrid_config_contains_prd_tracking():
    config = load_experiment_config("configs/experiments/hybrid/lgb_ranker.yaml")
    assert config.tracking.experiment_name == "movie-recs-prd"
    assert config.tracking.registered_model_name == "MovieRecs_EASE_LGB_Ranker"
    assert config.tracking.prd_tag == "PRD"
    assert config.training_params["candidate_top_k"] == 100
    assert config.training_params["features"] == PRD_FEATURES
    assert config.training_params["user_counter_features"]["enabled"] is True
    assert set(USER_COUNTER_FEATURES) <= set(config.training_params["features"])
    assert config.data.hybrid_table == "data/processed/hybrid_train.parquet"
    assert config.data.hybrid_test_table == "data/processed/hybrid_test.parquet"


def test_error_analysis_returns_representative_categories():
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2],
            "movieId": [10, 11, 12, 13, 20, 21, 22, 23],
            "title": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "label": [0, 1, 1, 0, 0, 1, 0, 1],
            "score": [0.9, 0.1, 0.2, 0.3, 0.8, 0.01, 0.2, 0.3],
            "ranker_score": [0.99, 0.4, 0.1, 0.2, 0.95, 0.05, 0.3, 0.2],
            "popularity": [0.95, 0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.1],
            "num_interactions": [20, 20, 20, 20, 3, 3, 3, 3],
            "main_genre": ["Drama"] * 8,
            "main_region": ["US"] * 8,
            "has_director": [1] * 8,
            "runtimeMinutes": [100] * 8,
            "rating": [7.0] * 8,
        }
    )
    errors = build_error_analysis(frame, "ranker_score", top_k=1, max_examples=10)
    assert {"false_positive_top_k", "false_negative_outside_top_k"} <= set(errors["error_type"])
    assert "popularity bias" in set(errors["category"])
    assert "weak user profile" in set(errors["category"])


def test_robustness_report_has_expected_scenarios():
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 11, 12, 20, 21, 22],
            "label": [1, 0, 0, 0, 1, 0],
            "score": [0.8, 0.3, 0.2, 0.2, 0.7, 0.1],
            "popularity": [0.2, 0.8, 0.1, 0.4, 0.3, 0.2],
            "age": [10, 20, 30, 15, 25, 35],
            "rating": [7.5, 6.0, 5.0, 6.5, 8.0, 5.5],
            "runtimeMinutes": [100, 90, 110, 105, 120, 95],
            "num_translations": [10, 2, 1, 5, 12, 3],
            "cluster": [1, 2, 3, 1, 2, 3],
            "has_director": [1, 1, 1, 1, 1, 1],
            "num_interactions": [30, 30, 30, 40, 40, 40],
            "main_genre": ["Drama", "Comedy", "Action", "Drama", "Drama", "Comedy"],
            "main_region": ["US"] * 6,
            "main_actor_popularity": [5, 3, 1, 4, 6, 2],
        }
    )

    def predict_scores(batch: pd.DataFrame):
        return batch["score"] + 0.1 * batch["rating"]

    report = run_feature_robustness(frame, PRD_FEATURES, predict_scores)
    assert set(report["scenario"]) == {"original", "numeric_noise_1pct", "metadata_degraded"}
    assert report["top_k_jaccard"].between(0, 1).all()


def test_user_counter_features_are_built_from_ratings_not_labels():
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 20],
            "label": [0, 1, 1],
            "main_genre": ["Drama", "Comedy", "Drama"],
            "main_region": ["US", "CA", "US"],
            "cluster": [1.0, 2.0, 1.0],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2],
            "movieId": [100, 101, 102, 200],
            "rating": [5.0, 2.0, 5.0, 4.0],
            "timestamp": [1500000000, 1500000001, 1600000000, 1500000002],
        }
    )
    item_metadata = pd.DataFrame(
        {
            "movieId": [10, 11, 20, 100, 101, 102, 200],
            "main_genre": ["Drama", "Comedy", "Drama", "Drama", "Comedy", "Drama", "Comedy"],
            "main_region": ["US", "CA", "US", "US", "CA", "US", "CA"],
            "cluster": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0],
        }
    )

    enriched = add_user_counter_features(frame, ratings, item_metadata, cutoff_date="2018-01-01", min_rating=3.0)
    relabeled = frame.copy()
    relabeled["label"] = 1 - relabeled["label"]
    enriched_relabeled = add_user_counter_features(relabeled, ratings, item_metadata, cutoff_date="2018-01-01", min_rating=3.0)

    user1_drama = enriched[(enriched["userId"] == 1) & (enriched["movieId"] == 10)].iloc[0]
    user1_comedy = enriched[(enriched["userId"] == 1) & (enriched["movieId"] == 11)].iloc[0]
    user2_drama = enriched[(enriched["userId"] == 2) & (enriched["movieId"] == 20)].iloc[0]

    assert user1_drama["user_main_genre_count"] == 1.0
    assert user1_drama["user_main_genre_share"] == 1.0
    assert user1_drama["user_main_region_count"] == 1.0
    assert user1_drama["user_cluster_count"] == 1.0
    assert user1_comedy["user_main_genre_count"] == 0.0
    assert user2_drama["user_main_genre_count"] == 0.0
    pd.testing.assert_frame_equal(
        enriched[USER_COUNTER_FEATURES],
        enriched_relabeled[USER_COUNTER_FEATURES],
    )


def test_user_counter_features_missing_ratings_error(tmp_path):
    config = load_experiment_config("configs/experiments/hybrid/lgb_ranker.yaml")
    config.data.ratings = str(tmp_path / "missing_ratings.csv")
    frame = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "label": [0],
            "main_genre": ["Drama"],
            "main_region": ["US"],
            "cluster": [1.0],
        }
    )

    with pytest.raises(FileNotFoundError, match="User counter features are enabled"):
        _apply_user_counter_features(config, frame, None)


def test_mlflow_dependency_smoke(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("movie-recs-prd-test")
    with mlflow.start_run() as run:
        mlflow.set_tag("stage", "PRD")
        mlflow.log_metric("test.ndcg@10", 0.5)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("movie-recs-prd-test")
    runs = client.search_runs([experiment.experiment_id], filter_string="tags.stage = 'PRD'")
    assert runs[0].info.run_id == run.info.run_id
