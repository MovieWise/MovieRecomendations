"""Typed configuration schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataPaths:
    """Locations of raw and derived datasets."""

    ratings: str = "data/raw/ratings.csv"
    movies: str = "data/raw/movies.csv"
    links: str = "data/raw/links.csv"
    tags: str = "data/raw/tags.csv"
    content: str = "data/processed/content.parquet"
    hybrid_table: str = "data/processed/hybrid_train.csv"
    hybrid_test_table: str = "data/processed/hybrid_test.parquet"
    artifacts_root: str = "artifacts"


@dataclass(slots=True)
class ArtifactBundle:
    """Structured output locations for one experiment."""

    root: str
    model_dir: str
    encoder_dir: str
    feature_dir: str
    metrics_dir: str
    plots_dir: str

    @classmethod
    def from_root(cls, root: str | Path) -> "ArtifactBundle":
        root_path = Path(root)
        return cls(
            root=str(root_path),
            model_dir=str(root_path / "model"),
            encoder_dir=str(root_path / "encoders"),
            feature_dir=str(root_path / "features"),
            metrics_dir=str(root_path / "metrics"),
            plots_dir=str(root_path / "plots"),
        )


@dataclass(slots=True)
class TrackingConfig:
    """Experiment tracking settings."""

    experiment_name: str = "movie-recs-prd"
    registered_model_name: str = "MovieRecs_EASE_LGB_Ranker"
    prd_tag: str = "PRD"
    tracking_uri: str | None = None


@dataclass(slots=True)
class ExperimentConfig:
    """High-level experiment configuration used by CLI entrypoints."""

    name: str
    family: str
    model_name: str
    seed: int = 42
    split_strategy: str = "temporal"
    split_params: dict[str, Any] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)
    training_params: dict[str, Any] = field(default_factory=dict)
    evaluation_params: dict[str, Any] = field(default_factory=dict)
    data: DataPaths = field(default_factory=DataPaths)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    artifacts: ArtifactBundle | None = None

    def with_default_artifacts(self) -> "ExperimentConfig":
        """Populate artifact paths when they are omitted."""
        if self.artifacts is None:
            root = Path(self.data.artifacts_root) / self.name
            self.artifacts = ArtifactBundle.from_root(root)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a plain dictionary."""
        return asdict(self)
