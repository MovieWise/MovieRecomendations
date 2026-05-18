import pandas as pd
from scipy.sparse import csr_matrix

from movie_recs.models.baselines import ModifiedTopPopularRecommender, TopPopularRecommender
from movie_recs.models.hybrid_rankers import LightGBMHybridRanker
from movie_recs.models.linear import EASERecommender
from movie_recs.models.matrix_factorization import PureSVDRecommender
from movie_recs.utils.io import save_pickle
from movie_recs.preprocessing.ranking_dataset import RankingDatasetBuilder


class FakeRawRanker:
    feature_name_ = ["score", "rating"]

    def predict(self, frame):
        return [0.5] * len(frame)


def test_top_popular_filters_seen_items():
    frame = pd.DataFrame(
        {
            "userId": [1, 2],
            "train_interactions": [[(10, 5), (11, 4)], [(10, 3), (12, 5)]],
        }
    )
    model = ModifiedTopPopularRecommender().fit(frame)
    recs = model.recommend(user_id=1, top_k=2)
    assert 10 not in recs
    assert 11 not in recs


def test_ease_and_puresvd_recommendations_exclude_seen():
    matrix = csr_matrix(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=float,
    )
    ease = EASERecommender(reg_weight=10.0).fit(matrix)
    svd = PureSVDRecommender(n_factors=2).fit(matrix)

    ease_recs = ease.recommend(0, top_k=2)
    svd_recs = svd.recommend(0, top_k=2)

    assert 0 not in ease_recs and 1 not in ease_recs
    assert 0 not in svd_recs and 1 not in svd_recs


def test_ranking_dataset_builder():
    frame = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "genres": ["['Comedy', 'Romance']"],
            "actors_list": ["['A', 'B']"],
            "director_name": ["D"],
            "age": [None],
            "runtimeMinutes": [90.0],
            "num_translations": [None],
            "cluster": [None],
            "main_region": [None],
            "is_multiregional": [1.0],
        }
    )
    result = RankingDatasetBuilder().build(frame, truth_items={1: {10}})
    assert result.loc[0, "main_genre"] == "Comedy"
    assert result.loc[0, "label"] == 1


def test_lightgbm_ranker_load_wraps_raw_predictor(tmp_path):
    path = tmp_path / "raw_ranker.pkl"
    save_pickle(FakeRawRanker(), path)
    ranker = LightGBMHybridRanker.load(path)
    assert ranker.model is not None
    assert ranker.feature_names == ["score", "rating"]
