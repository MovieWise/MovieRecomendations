import pytest


def test_ncf_forward_shape():
    torch = pytest.importorskip("torch")
    from movie_recs.models.deep.ncf import NCF

    model = NCF(n_users=3, m_items=5, n_factors=4, hidden_dim=8)
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([1, 2, 3], dtype=torch.long)
    scores = model(items, users)
    assert scores.shape == (3,)


def test_bert4rec_dataset_and_collate():
    pytest.importorskip("torch")
    from movie_recs.models.deep.bert4rec import BERT4RecDataset, collate_fn_test, collate_fn_train
    import pandas as pd

    joined = pd.DataFrame(
        {
            "train_interactions": [[(10, 1), (11, 1)], [(11, 1)]],
            "test_interactions": [[(12, 1)], [(10, 1)]],
        }
    )
    item2idx = {10: 1, 11: 2, 12: 3}
    train_dataset = BERT4RecDataset(joined, item2idx, max_len=5, phase="train")
    test_dataset = BERT4RecDataset(joined, item2idx, max_len=5, phase="test")

    train_batch = collate_fn_train([train_dataset[0], train_dataset[1]])
    test_batch = collate_fn_test([test_dataset[0], test_dataset[1]])

    assert train_batch["seq_i"].shape[0] == 2
    assert test_batch["seq_i"].shape[0] == 2
    assert len(test_batch["targets"]) == 2
