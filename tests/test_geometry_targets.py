import torch

from uvnet_retrieval.geometry_targets import compute_geometry_targets


def test_geometry_targets_shape_and_zero_curvature():
    normals = torch.tensor(
        [[[[0.0, 0.0, 1.0],
           [0.0, 0.0, 1.0]],
          [[0.0, 0.0, 1.0],
           [0.0, 0.0, 1.0]]]]
    )
    points = torch.zeros(1, 2, 2, 3)
    mask = torch.ones(1, 2, 2, 1)
    feat = torch.cat([points, normals, mask], dim=-1)

    out = compute_geometry_targets(feat)
    assert out.shape == (1, 6)
    assert torch.allclose(out, torch.zeros_like(out))
