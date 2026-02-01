import numpy as np
from uvnet_retrieval.masking import random_mask

def test_random_mask_ratio():
    mask = random_mask(100, mask_ratio=0.6, seed=0)
    assert mask.sum() == 60
