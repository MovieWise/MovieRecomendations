"""Training entrypoints for the different model families."""

from __future__ import annotations

from typing import Any

from movie_recs.config.schemas import ExperimentConfig
from movie_recs.models.baselines import ModifiedTopPopularRecommender, RandomRecommender, TopPopularRecommender
from movie_recs.models.linear import EASERecommender, SLIMRecommender
from movie_recs.models.matrix_factorization import PureSVDRecommender
from movie_recs.models.neighbors import ItemKNNRecommender, UserKNNRecommender
from movie_recs.training.artifacts import save_artifact_bundle


def build_model_from_config(config: ExperimentConfig) -> Any:
    """Instantiate a model from the high-level experiment config."""
    params = config.model_params
    registry = {
        "random": lambda: RandomRecommender(),
        "toppop": lambda: TopPopularRecommender(),
        "modified_toppop": lambda: ModifiedTopPopularRecommender(),
        "ease": lambda: EASERecommender(**params),
        "slim": lambda: SLIMRecommender(**params),
        "puresvd": lambda: PureSVDRecommender(**params),
        "item_knn": lambda: ItemKNNRecommender(**params),
        "user_knn": lambda: UserKNNRecommender(**params),
        "ncf": lambda: __import__("movie_recs.models.deep.ncf", fromlist=["NCF"]).NCF(
            n_users=params["n_users"],
            m_items=params["m_items"],
            n_factors=params.get("n_factors", 32),
            hidden_dim=params.get("hidden_dim", 128),
        ),
        "bert4rec": lambda: __import__("movie_recs.models.deep.bert4rec", fromlist=["BERT4Rec"]).BERT4Rec(
            n_items=params["n_items"],
            max_seq_length=params.get("max_seq_length", 100),
        ),
    }
    try:
        return registry[config.model_name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported model_name: {config.model_name}") from exc


def train_from_config(config: ExperimentConfig, train_data: Any, *fit_args, **fit_kwargs) -> Any:
    """Build a model, fit it and initialize artifact directories."""
    config = config.with_default_artifacts()
    model = build_model_from_config(config)
    model.fit(train_data, *fit_args, **fit_kwargs)
    save_artifact_bundle(
        config.artifacts,
        metadata={
            "name": config.name,
            "family": config.family,
            "model_name": config.model_name,
            "model_params": config.model_params,
            "training_params": config.training_params,
        },
    )
    return model
