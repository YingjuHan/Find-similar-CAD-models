import numpy as np
from cad_similarity.similarity import correlation_similarity, cluster_by_threshold

def test_correlation_similarity_identity():
    a = np.eye(3)
    b = np.eye(3)
    sim = correlation_similarity(a, b)
    assert np.isclose(sim, 1.0)


def test_cluster_by_threshold():
    sims = [1.0, 0.995, 0.98, 0.975]
    clusters = cluster_by_threshold(sims, threshold=0.01)
    assert clusters == [[0, 1], [2, 3]]
