import numpy as np
from cad_similarity.local_similarity import local_similarity


def test_local_similarity_identical():
    mats_a = [np.eye(4), np.eye(4)]
    mats_b = [np.eye(4), np.eye(4)]
    sim = local_similarity(mats_a, mats_b)
    assert np.isclose(sim, 1.0)
