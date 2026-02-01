import torch
import torch.nn.functional as F


def embedding_mse(student, teacher):
    return F.mse_loss(student, teacher)


def masked_mse(student, teacher, mask):
    if mask.numel() == 0:
        return torch.tensor(0.0, device=student.device)
    return F.mse_loss(student[mask], teacher[mask])
