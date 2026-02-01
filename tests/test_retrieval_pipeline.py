import numpy as np

from uvnet_retrieval.build_index import build_index


def test_build_index_fallback():
    emb = np.random.randn(5, 8).astype("float32")
    index = build_index(emb)
    assert index is not None
