import torch
from torch import nn


def prep_face_feat(x):
    if x.dim() != 4:
        raise ValueError("Expected face features (F,H,W,C) or (F,C,H,W)")
    if x.size(1) == 7:
        return x
    if x.size(-1) == 7:
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError("Invalid face feature layout")


def prep_edge_feat(x):
    if x.dim() != 3:
        raise ValueError("Expected edge features (E,U,C) or (E,C,U)")
    if x.size(1) == 6:
        return x
    if x.size(-1) == 6:
        return x.permute(0, 2, 1).contiguous()
    raise ValueError("Invalid edge feature layout")


class UVNetEncoder(nn.Module):
    def __init__(self, node_embed_dim=64, edge_embed_dim=64, graph_embed_dim=128):
        super().__init__()
        from uvnet.encoders import UVNetSurfaceEncoder, UVNetCurveEncoder, UVNetGraphEncoder

        self.surface = UVNetSurfaceEncoder(in_channels=7, output_dims=node_embed_dim)
        self.curve = UVNetCurveEncoder(in_channels=6, output_dims=edge_embed_dim)
        self.graph = UVNetGraphEncoder(
            input_dim=node_embed_dim,
            input_edge_dim=edge_embed_dim,
            output_dim=graph_embed_dim,
        )

    def forward(self, graph):
        x = prep_face_feat(graph.ndata["x"].float())
        e = prep_edge_feat(graph.edata["x"].float())
        node_feat = self.surface(x)
        edge_feat = self.curve(e)
        node_emb, graph_emb = self.graph(graph, node_feat, edge_feat)
        return node_emb, graph_emb
