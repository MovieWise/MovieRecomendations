"""Reusable content preprocessing logic extracted from the EDA notebooks."""

from __future__ import annotations

import ast

import pandas as pd


def parse_list_string(value: object, default: str = "Unknown") -> str:
    """Parse the first item from a stringified list."""
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        return parsed[0] if parsed else default
    except (ValueError, SyntaxError, TypeError):
        return default


def process_movie_group(group: pd.DataFrame) -> pd.Series:
    """Aggregate actor/director metadata per movie."""
    actors = group[group["category"].isin(["actor", "actress"])]["primaryName"].unique().tolist()
    director = group[group["category"] == "director"]["primaryName"].unique()
    result = group.iloc[0].copy()
    result["actors_list"] = actors
    result["director_name"] = director[0] if len(director) > 0 else None
    return result


def merge_movie_metadata(movies: pd.DataFrame, principals: pd.DataFrame) -> pd.DataFrame:
    """Merge principal cast/crew metadata into the movie table."""
    final_df = principals.groupby("movieId").apply(process_movie_group).reset_index(drop=True)
    drop_cols = [col for col in ["category", "primaryName", "star_power"] if col in final_df.columns]
    final_df = final_df.drop(columns=drop_cols)
    if "is_multiregional" in final_df.columns:
        final_df["is_multiregional"] = final_df["is_multiregional"].astype(bool)
    return movies.merge(final_df, on="movieId", how="left")

