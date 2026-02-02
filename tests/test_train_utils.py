from uvnet_retrieval.train_mae import sample_indices


def test_sample_indices_cap():
    idx = list(range(10))
    out = sample_indices(idx, max_items=4, seed=0)
    assert len(out) == 4
    assert sorted(out) == sorted(set(out))
