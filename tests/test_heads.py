import torch
from uvnet_retrieval.heads import GeometryHead, FaceTypeHead

def test_heads_shapes():
    x = torch.randn(5, 16)
    geo = GeometryHead(16, out_dim=6)
    face = FaceTypeHead(16, num_classes=5)
    g = geo(x)
    f = face(x)
    assert g.shape == (5, 6)
    assert f.shape == (5, 5)
