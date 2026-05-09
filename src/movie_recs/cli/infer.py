"""CLI for inference."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for MovieRecomendations models.")
    parser.add_argument("--model-path", required=True, help="Path to a serialized model artifact.")
    parser.add_argument("--user-id", type=int, help="User ID for known-user inference.")
    parser.add_argument("--item-ids", nargs="*", type=int, default=[], help="Movie IDs for profile-based inference.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate CLI arguments.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Model path: {args.model_path}")
    if args.dry_run:
        print("Dry run complete.")
        return 0
    if args.user_id is None and not args.item_ids:
        raise ValueError("Provide either --user-id or --item-ids.")
    print("Inference entrypoint scaffold is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
