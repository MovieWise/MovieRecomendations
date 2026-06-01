from pathlib import Path

from movie_recs.config.schemas import ArtifactBundle
from movie_recs.training.artifacts import save_artifact_bundle


def test_artifact_bundle_creation(tmp_path: Path):
    bundle = ArtifactBundle.from_root(tmp_path / "exp")
    save_artifact_bundle(bundle, {"status": "ok"})
    assert Path(bundle.model_dir).exists()
    assert (tmp_path / "exp" / "run_metadata.json").exists()

