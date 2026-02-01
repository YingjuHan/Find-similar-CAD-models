import numpy as np
import torch
from uvnet_retrieval.masking import random_mask, batch_random_mask, apply_node_mask

def test_random_mask_ratio():
    mask = random_mask(100, mask_ratio=0.6, seed=0)
    assert mask.sum() == 60


def test_batch_random_mask_counts():
    mask = batch_random_mask([10, 20], mask_ratio=0.5, seed=0)
    assert mask.shape[0] == 30
    assert mask[:10].sum() == 5
    assert mask[10:].sum() == 10


def test_apply_node_mask_zeroes():
    feats = torch.ones(4, 2)
    mask = torch.tensor([True, False, True, False])
    out = apply_node_mask(feats.clone(), mask)
    assert torch.all(out[mask] == 0)
    assert torch.all(out[~mask] == 1)
