import pandas as pd

from movie_recs.data.builders import build_interaction_matrix, encode_interactions
from movie_recs.data.splitters import group_split, temporal_split


def make_frame():
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 11, 10, 12, 11, 13],
            "rating": [5, 4, 3, 2, 4, 5],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )


def test_temporal_split_and_encoding():
    frame = make_frame()
    train_df, test_df = temporal_split(frame, cutoff=4)
    assert len(train_df) == 3
    assert len(test_df) == 3

    encoded = encode_interactions(train_df)
    matrix = build_interaction_matrix(encoded.frame)

    assert matrix.shape == (2, 3)
    assert encoded.user_encoder.inverse_transform([0, 1]).tolist() == [1, 2]


def test_group_split_is_reproducible():
    frame = make_frame()
    train_a, test_a = group_split(frame, random_state=42)
    train_b, test_b = group_split(frame, random_state=42)
    assert train_a.index.tolist() == train_b.index.tolist()
    assert test_a.index.tolist() == test_b.index.tolist()

