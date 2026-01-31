import numpy as np
from cad_similarity.synthetic import generate_parts, sample_part

def test_generate_parts_groups():
    parts = generate_parts(seed=0)
    assert len(parts) == 6
    groups = {p.group for p in parts}
    assert groups == {"rect", "tri", "arc"}


def test_sample_part_counts():
    parts = generate_parts(seed=0)
    pts, normals, labels = sample_part(parts[0], n_base=100, n_teeth=50, seed=0)
    assert pts.shape == (150, 3)
    assert normals.shape == (150, 3)
    assert labels.shape == (150,)
    assert labels.sum() == 50
