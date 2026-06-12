"""CLI for model training."""

from __future__ import annotations

import argparse

from movie_recs.config.loader import load_experiment_config
from movie_recs.training.hybrid_mlflow import train_prd_hybrid_ranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MovieRecomendations model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate the config and chosen model.")
    parser.add_argument("--mlflow", action="store_true", help="Log the final experiment to MLflow.")
    parser.add_argument("--limit-users", type=int, default=None, help="Train on the first N users for a quick smoke-test.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    print(f"Training config: {config.name} [{config.family}/{config.model_name}]")
    if args.dry_run:
        print("Dry run complete.")
        return 0
    if config.family == "hybrid" and config.name == "ease_lgb_ranker":
        result = train_prd_hybrid_ranker(
            config,
            log_to_mlflow=args.mlflow,
            limit_users=args.limit_users,
        )
        print(f"Run name: {result.run_name}")
        print(f"Artifacts: {result.artifact_dir}")
        for name, value in sorted(result.metrics.items()):
            print(f"{name}: {value}")
        return 0
    print("Training pipeline requires project-local datasets and is ready to be wired into experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
