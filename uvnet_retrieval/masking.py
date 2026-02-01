import numpy as np

def random_mask(n_tokens, mask_ratio=0.6, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_tokens, dtype=bool)
    idx = rng.choice(n_tokens, int(n_tokens * mask_ratio), replace=False)
    mask[idx] = True
    return mask
