import torch
import torch.nn as nn

class GeometryHead(nn.Module):
    def __init__(self, dim, out_dim=6):
        super().__init__()
        self.net = nn.Linear(dim, out_dim)
    def forward(self, x):
        return self.net(x)

class FaceTypeHead(nn.Module):
    def __init__(self, dim, num_classes=5):
        super().__init__()
        self.net = nn.Linear(dim, num_classes)
    def forward(self, x):
        return self.net(x)
