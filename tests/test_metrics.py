import math

from movie_recs.metrics.ranking import dcg, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics_on_small_fixture():
    gt_items = {10, 20}
    predicted = [10, 30, 20]

    assert math.isclose(dcg([1, 0, 1]), 1.5, rel_tol=1e-6)
    assert math.isclose(ndcg_at_k(gt_items, predicted, k=3), 0.9197207891481876, rel_tol=1e-6)
    assert precision_at_k(gt_items, predicted, k=2) == 0.5
    assert recall_at_k(gt_items, predicted, k=3) == 1.0

