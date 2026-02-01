import torch
from uvnet_retrieval.losses import embedding_mse

def test_embedding_mse_zero():
    x = torch.zeros(4, 8)
    y = torch.zeros(4, 8)
    assert embedding_mse(x, y).item() == 0.0
