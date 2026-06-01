"""CLI for data preparation."""

from __future__ import annotations

import argparse

from movie_recs.config.loader import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare datasets for MovieRecomendations experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate the config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    print(f"Loaded config: {config.name} [{config.family}/{config.model_name}]")
    if args.dry_run:
        print("Dry run complete.")
        return 0
    print("Data preparation pipeline scaffold is ready for project-specific dataset wiring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

