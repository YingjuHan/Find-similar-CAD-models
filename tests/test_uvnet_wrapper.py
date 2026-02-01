import pytest
import torch

from uvnet_retrieval.uvnet_wrapper import UVNetEncoder, prep_edge_feat, prep_face_feat


def test_uvnet_wrapper_importable():
    assert UVNetEncoder is not None


def test_uvnet_wrapper_prep_layouts():
    face = torch.randn(2, 10, 10, 7)
    edge = torch.randn(3, 10, 6)
    face_c = prep_face_feat(face)
    edge_c = prep_edge_feat(edge)
    assert face_c.shape == (2, 7, 10, 10)
    assert edge_c.shape == (3, 6, 10)


def test_uvnet_wrapper_forward_shapes():
    dgl = pytest.importorskip("dgl")
    g = dgl.graph(([0], [1]), num_nodes=2)
    g.ndata["x"] = torch.randn(2, 10, 10, 7)
    g.edata["x"] = torch.randn(1, 10, 6)
    model = UVNetEncoder(node_embed_dim=64, edge_embed_dim=64, graph_embed_dim=128)
    node_emb, graph_emb = model(g)
    assert node_emb.shape[0] == 2
    assert graph_emb.shape[-1] == 128
