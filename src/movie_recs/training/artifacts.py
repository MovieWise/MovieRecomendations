"""Artifact persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from movie_recs.config.schemas import ArtifactBundle
from movie_recs.utils.io import ensure_dir, save_json


def save_artifact_bundle(bundle: ArtifactBundle, metadata: dict[str, Any]) -> ArtifactBundle:
    """Create bundle directories and persist run metadata."""
    for directory in [
        bundle.root,
        bundle.model_dir,
        bundle.encoder_dir,
        bundle.feature_dir,
        bundle.metrics_dir,
        bundle.plots_dir,
    ]:
        ensure_dir(directory)
    save_json(metadata, Path(bundle.root) / "run_metadata.json")
    return bundle

