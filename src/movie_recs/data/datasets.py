"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_interactions(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load MovieLens-style interactions from CSV."""
    return pd.read_csv(path, usecols=list(usecols) if usecols else None)


def load_content_frame(path: str | Path) -> pd.DataFrame:
    """Load content metadata from CSV or parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

