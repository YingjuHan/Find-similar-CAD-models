import tempfile
from pathlib import Path

import torch
import pytest


def test_extract_embeddings_writes_file():
    dgl = pytest.importorskip("dgl")
    from dgl.data.utils import save_graphs
    from uvnet_retrieval.extract_embeddings import extract_embeddings

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        g = dgl.graph(([0], [1]), num_nodes=2)
        g.ndata["x"] = torch.zeros(2, 10, 10, 7)
        g.edata["x"] = torch.zeros(1, 10, 6)
        save_graphs(str(root / "a.bin"), [g])
        out = root / "emb.pt"
        extract_embeddings(
            graph_root=str(root),
            checkpoint=None,
            out_path=str(out),
            device="cpu",
        )
        assert out.exists()
