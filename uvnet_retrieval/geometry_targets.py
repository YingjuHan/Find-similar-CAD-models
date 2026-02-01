import torch


def _to_hwc(x):
    if x.dim() != 4:
        raise ValueError("Expected (F,H,W,C) or (F,C,H,W)")
    if x.size(-1) == 7:
        return x
    if x.size(1) == 7:
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError("Could not infer UV-grid channel layout")


def compute_geometry_targets(face_feat):
    feat = _to_hwc(face_feat)
    normals = feat[..., 3:6]
    mask = feat[..., 6:7]
    mask = (mask > 0).float()

    n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    masked_n = n * mask
    mean_n = masked_n.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
    mean_n = mean_n / (mean_n.norm(dim=-1, keepdim=True) + 1e-8)

    dot = (n * mean_n[:, None, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(dot) * mask.squeeze(-1)
    valid = mask.squeeze(-1)
    angle_mean = angles.sum(dim=(1, 2)) / (valid.sum(dim=(1, 2)) + 1e-8)
    angle_var = ((angles - angle_mean[:, None, None]) ** 2 * valid).sum(dim=(1, 2))
    angle_var = angle_var / (valid.sum(dim=(1, 2)) + 1e-8)
    angle_std = torch.sqrt(torch.clamp(angle_var, min=0.0))

    du = n[:, 1:, :, :] - n[:, :-1, :, :]
    dv = n[:, :, 1:, :] - n[:, :, :-1, :]
    du_mag = du.norm(dim=-1)
    dv_mag = dv.norm(dim=-1)
    du_mean = du_mag.mean(dim=(1, 2))
    du_std = du_mag.std(dim=(1, 2), unbiased=False)
    dv_mean = dv_mag.mean(dim=(1, 2))
    dv_std = dv_mag.std(dim=(1, 2), unbiased=False)

    return torch.stack([angle_mean, angle_std, du_mean, du_std, dv_mean, dv_std], dim=-1)
