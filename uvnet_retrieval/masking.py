import numpy as np
import torch


def _mask_count(n_tokens, mask_ratio):
    return max(1, int(n_tokens * mask_ratio)) if n_tokens > 0 else 0


def random_mask(n_tokens, mask_ratio=0.6, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_tokens, dtype=bool)
    count = _mask_count(n_tokens, mask_ratio)
    if count == 0:
        return mask
    idx = rng.choice(n_tokens, count, replace=False)
    mask[idx] = True
    return mask


def batch_random_mask(batch_sizes, mask_ratio=0.6, seed=0):
    rng = np.random.default_rng(seed)
    masks = []
    for n in batch_sizes:
        mask = np.zeros(n, dtype=bool)
        count = _mask_count(n, mask_ratio)
        if count > 0:
            idx = rng.choice(n, count, replace=False)
            mask[idx] = True
        masks.append(mask)
    if not masks:
        return np.zeros(0, dtype=bool)
    return np.concatenate(masks)


def apply_node_mask(features, mask, mask_token=None):
    if not torch.is_tensor(features):
        raise TypeError("features must be a torch.Tensor")
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=features.device)
    if mask_token is None:
        features[mask] = 0
        return features
    token = mask_token.to(features.device)
    if token.dim() == features.dim() - 1:
        token = token.unsqueeze(0)
    features[mask] = token
    return features
