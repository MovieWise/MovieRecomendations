from pathlib import Path

from movie_recs.cli.evaluate import main as evaluate_main
from movie_recs.cli.prepare_data import main as prepare_main
from movie_recs.cli.train import main as train_main
from movie_recs.config.loader import load_experiment_config


def test_load_config_and_cli_smoke():
    config_path = Path("configs/experiments/base/ease.yaml")
    config = load_experiment_config(config_path)
    assert config.name == "ease_baseline"
    assert config.artifacts is not None

    assert prepare_main(["--config", str(config_path), "--dry-run"]) == 0
    assert train_main(["--config", str(config_path), "--dry-run"]) == 0
    assert evaluate_main(["--config", str(config_path), "--dry-run"]) == 0

