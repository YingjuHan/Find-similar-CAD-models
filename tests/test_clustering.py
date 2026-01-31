import numpy as np
from cad_similarity.clustering import dbscan_clusters


def test_dbscan_finds_two_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.02, size=(30, 3))
    b = rng.normal(loc=1.0, scale=0.02, size=(30, 3))
    pts = np.vstack([a, b])
    labels = dbscan_clusters(pts, eps=0.08, min_samples=5)
    found = set(labels)
    found.discard(-1)
    assert len(found) >= 2
