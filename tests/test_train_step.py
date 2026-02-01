import pytest
import torch

from uvnet_retrieval.train_mae import compute_losses


def test_compute_losses_shapes():
    dgl = pytest.importorskip("dgl")
    g = dgl.graph(([0], [1]), num_nodes=2)
    g.ndata["x"] = torch.randn(2, 10, 10, 7)
    g.edata["x"] = torch.randn(1, 10, 6)
    mask = torch.tensor([True, False])
    loss = compute_losses(g, mask)
    assert loss.item() >= 0.0
