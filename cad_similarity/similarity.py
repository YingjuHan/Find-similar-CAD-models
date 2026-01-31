import numpy as np


def correlation_similarity(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    c = np.corrcoef(a, b)
    return float(c[0, 1])


def cluster_by_threshold(similarities, threshold=0.01):
    clusters = []
    current = []
    for idx, val in enumerate(similarities):
        if not current:
            current = [idx]
            last_val = val
            continue
        if abs(last_val - val) <= threshold:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
        last_val = val
    if current:
        clusters.append(current)
    return clusters
