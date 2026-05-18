from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommendation_service.schemas.api import MovieInfo


class MovieCatalogService:
    def __init__(self, movies_path: str, links_path: str, content_features_path: str) -> None:
        self.movies_path = Path(movies_path)
        self.links_path = Path(links_path)
        self.content_features_path = Path(content_features_path)
        self._movies: pd.DataFrame | None = None
        self._content: pd.DataFrame | None = None

    @property
    def movies(self) -> pd.DataFrame:
        if self._movies is None:
            self._movies = self._load_movies()
        return self._movies

    @property
    def content(self) -> pd.DataFrame:
        if self._content is None:
            self._content = self._load_content()
        return self._content

    def _load_movies(self) -> pd.DataFrame:
        movies = pd.DataFrame(columns=["movieId", "title"])
        links = pd.DataFrame(columns=["movieId", "imdbId"])
        if self.movies_path.exists():
            movies = pd.read_csv(self.movies_path)
        if self.links_path.exists():
            links = pd.read_csv(self.links_path)
        elif self.movies_path.exists() and {"movieId", "imdbId"}.issubset(movies.columns):
            links = movies[["movieId", "imdbId"]].copy()

        if movies.empty and {"movieId", "title"}.issubset(links.columns):
            movies = links[["movieId", "title"]].copy()
        elif "movieId" not in movies.columns and "movieId" in links.columns:
            movies = links[["movieId"]].copy()
        if "movieId" in movies.columns and "movieId" in links.columns:
            link_cols = [
                col for col in ["movieId", "imdbId", "tmdbId"]
                if col in links.columns and (col == "movieId" or col not in movies.columns)
            ]
            result = movies.merge(links[link_cols], on="movieId", how="left") if len(link_cols) > 1 else movies
        else:
            result = movies
        if "movieId" in result.columns:
            result["movieId"] = result["movieId"].astype(int)
        return result

    def _load_content(self) -> pd.DataFrame:
        if not self.content_features_path.exists():
            return pd.DataFrame()
        if self.content_features_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(self.content_features_path)
        else:
            frame = pd.read_csv(self.content_features_path)
        if "movieId" in frame.columns:
            frame["movieId"] = frame["movieId"].astype(int)
        return frame

    def is_available(self) -> bool:
        return not self.movies.empty

    def get_movie(self, movie_id: int) -> dict | None:
        if self.movies.empty or "movieId" not in self.movies.columns:
            return None
        row = self.movies.loc[self.movies["movieId"] == int(movie_id)]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_imdb_id(self, movie_id: int) -> str | None:
        movie = self.get_movie(movie_id)
        if not movie:
            return None
        imdb_id = movie.get("imdbId")
        if pd.isna(imdb_id):
            return None
        imdb_text = str(int(imdb_id)) if isinstance(imdb_id, float) else str(imdb_id)
        if imdb_text.startswith("tt"):
            return imdb_text
        return f"tt{imdb_text.zfill(7)}"

    def get_base_info(self, movie_id: int) -> MovieInfo:
        movie = self.get_movie(movie_id) or {}
        title = movie.get("title") or movie.get("Title")
        return MovieInfo(
            movie_id=int(movie_id),
            imdb_id=self.get_imdb_id(movie_id),
            title=None if pd.isna(title) else title,
            imdb_url=f"https://www.imdb.com/title/{self.get_imdb_id(movie_id)}/" if self.get_imdb_id(movie_id) else None,
        )

    def feed(self, excluded_movie_ids: set[int], limit: int) -> list[int]:
        if self.movies.empty or "movieId" not in self.movies.columns:
            return []
        movie_ids = self._default_feed_movie_ids()
        result: list[int] = []
        for movie_id in movie_ids:
            if movie_id in excluded_movie_ids:
                continue
            result.append(movie_id)
            if len(result) >= limit:
                break
        return result

    def _default_feed_movie_ids(self) -> list[int]:
        if not self.content.empty and {"movieId", "popularity"}.issubset(self.content.columns):
            return (
                self.content[["movieId", "popularity"]]
                .dropna(subset=["movieId"])
                .sort_values(["popularity", "movieId"], ascending=[False, True])
                ["movieId"]
                .astype(int)
                .tolist()
            )
        return self.movies["movieId"].dropna().astype(int).tolist()

    def search(self, query: str, excluded_movie_ids: set[int] | None = None, limit: int = 20) -> list[int]:
        if self.movies.empty or "movieId" not in self.movies.columns:
            return []
        excluded = excluded_movie_ids or set()
        normalized_query = query.strip().casefold()
        if not normalized_query:
            return self.feed(excluded, limit)
        title_col = "title" if "title" in self.movies.columns else "Title"
        if title_col not in self.movies.columns:
            return []
        mask = self.movies[title_col].fillna("").str.casefold().str.contains(normalized_query, regex=False)
        result: list[int] = []
        for movie_id in self.movies.loc[mask, "movieId"].dropna().astype(int).tolist():
            if movie_id in excluded:
                continue
            result.append(movie_id)
            if len(result) >= limit:
                break
        return result

    def build_candidate_features(self, candidates: list[tuple[int, float]]) -> pd.DataFrame:
        frame = pd.DataFrame(candidates, columns=["movieId", "score"])
        if frame.empty:
            return frame
        if not self.content.empty and "movieId" in self.content.columns:
            frame = frame.merge(self.content, on="movieId", how="left")
        if "title" not in frame.columns and "movieId" in self.movies.columns:
            frame = frame.merge(self.movies[["movieId", "title"]], on="movieId", how="left")
        return frame
