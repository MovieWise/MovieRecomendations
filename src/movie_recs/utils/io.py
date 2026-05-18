"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Persist a Python object with pickle."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("wb") as file_obj:
        pickle.dump(obj, file_obj)
    return target


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file."""
    with Path(path).open("rb") as file_obj:
        return pickle.load(file_obj)


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist an object with joblib."""
    target = Path(path)
    ensure_dir(target.parent)
    joblib.dump(obj, target)
    return target


def load_joblib(path: str | Path) -> Any:
    """Load a joblib file."""
    return joblib.load(Path(path))


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Persist a JSON-serializable dictionary."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    return target

