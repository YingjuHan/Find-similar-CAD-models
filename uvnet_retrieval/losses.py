import torch
import torch.nn.functional as F


def embedding_mse(student, teacher):
    return F.mse_loss(student, teacher)
