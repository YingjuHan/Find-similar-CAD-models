import numpy as np
from cad_similarity.projection import spherical_projection

def test_spherical_projection_shape_and_sum():
    dirs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    mat = spherical_projection(dirs, bins=(10, 10))
    assert mat.shape == (10, 10)
    assert np.isclose(mat.sum(), 1.0)
