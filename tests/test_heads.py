import torch
from uvnet_retrieval.heads import GeometryHead, FaceTypeHead

def test_heads_shapes():
    x = torch.randn(5, 16)
    geo = GeometryHead(16)
    face = FaceTypeHead(16, num_classes=5)
    g = geo(x)
    f = face(x)
    assert g.shape == (5, 4)
    assert f.shape == (5, 5)
