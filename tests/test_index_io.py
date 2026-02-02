import tempfile
from pathlib import Path

import numpy as np

from uvnet_retrieval.build_index import build_index, load_index, save_index


def test_index_roundtrip_numpy():
    emb = np.random.randn(4, 8).astype("float32")
    index = build_index(emb)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_index(index, emb, root)
        loaded = load_index(root)
        assert loaded is not None
