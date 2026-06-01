import asyncio

from recommendation_service.infrastructure import database as db
from recommendation_service.repositories.omdb_cache import OmdbCacheRepository
from recommendation_service.services.catalog import MovieCatalogService
from recommendation_service.services.omdb_service import OmdbService


class FakeClient:
    async def fetch_by_imdb_id(self, imdb_id):
        return {
            "Title": "Heat",
            "Year": "1995",
            "Genre": "Crime",
            "Plot": "A professional thief crosses paths with a detective.",
            "Poster": "N/A",
            "imdbRating": "8.3",
            "Runtime": "170 min",
            "Director": "Michael Mann",
            "Actors": "Al Pacino, Robert De Niro",
            "imdbID": imdb_id,
            "Response": "True",
        }


def test_omdb_service_caches_response(tmp_path):
    movies = tmp_path / "movies.csv"
    links = tmp_path / "links.csv"
    movies.write_text("movieId,title\n6,Heat\n", encoding="utf-8")
    links.write_text("movieId,imdbId\n6,113277\n", encoding="utf-8")
    db.init_db(f"sqlite:///{tmp_path / 'cache.db'}")
    session = next(db.get_db())
    try:
        service = OmdbService(
            FakeClient(),
            OmdbCacheRepository(session),
            MovieCatalogService(str(movies), str(links), str(tmp_path / "missing.parquet")),
            ttl_seconds=60,
        )
        movie = asyncio.run(service.get_movie_info(6))
        assert movie.title == "Heat"
        assert movie.poster is None
        cached = OmdbCacheRepository(session).get_fresh(6)
        assert cached is not None
    finally:
        session.close()
