import numpy as np


def spherical_projection(directions, bins=(224, 224)):
    directions = np.asarray(directions, dtype=float)
    # Normalize defensively
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms != 0)

    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arctan2(y, x)  # [-pi, pi]
    phi = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]

    h, w = bins
    u = ((theta + np.pi) / (2 * np.pi) * w).astype(int)
    v = ((phi) / np.pi * h).astype(int)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    mat = np.zeros((h, w), dtype=float)
    for ui, vi in zip(u, v):
        mat[vi, ui] += 1.0

    total = mat.sum()
    if total > 0:
        mat /= total
    return mat
