import tempfile
from pathlib import Path

import numpy as np

from uvnet_retrieval.build_index import build_index, save_index
from uvnet_retrieval.search import search_index


def test_search_index_returns_k():
    emb = np.random.randn(5, 8).astype("float32")
    index = build_index(emb)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_index(index, emb, root)
        idx, dist = search_index(root, emb[0], top_k=3)
        assert len(idx) == 3
        assert len(dist) == 3
