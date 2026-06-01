"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from movie_recs.config.schemas import ArtifactBundle, DataPaths, ExperimentConfig, TrackingConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return payload


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    payload = _read_yaml(path)
    data_payload = payload.get("data", {})
    tracking_payload = payload.get("tracking", {})
    artifacts_payload = payload.get("artifacts")

    config = ExperimentConfig(
        name=payload["name"],
        family=payload["family"],
        model_name=payload["model_name"],
        seed=payload.get("seed", 42),
        split_strategy=payload.get("split_strategy", "temporal"),
        split_params=payload.get("split_params", {}),
        model_params=payload.get("model_params", {}),
        training_params=payload.get("training_params", {}),
        evaluation_params=payload.get("evaluation_params", {}),
        data=DataPaths(**data_payload),
        tracking=TrackingConfig(**tracking_payload),
        artifacts=ArtifactBundle(**artifacts_payload) if artifacts_payload else None,
    )
    return config.with_default_artifacts()
