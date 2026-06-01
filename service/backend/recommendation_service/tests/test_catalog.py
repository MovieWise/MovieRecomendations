import pandas as pd

from recommendation_service.services.catalog import MovieCatalogService


def test_feed_uses_content_popularity(tmp_path):
    movies = tmp_path / "links.csv"
    content = tmp_path / "content.parquet"
    movies.write_text(
        "movieId,imdbId,title\n1,114709,Toy Story\n2,113497,Jumanji\n3,113228,Grumpier Old Men\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "popularity": [0.2, 0.9, 0.5],
        }
    ).to_parquet(content)

    catalog = MovieCatalogService(str(movies), str(movies), str(content))

    assert catalog.feed(excluded_movie_ids=set(), limit=3) == [2, 3, 1]
    assert catalog.feed(excluded_movie_ids={2}, limit=2) == [3, 1]

