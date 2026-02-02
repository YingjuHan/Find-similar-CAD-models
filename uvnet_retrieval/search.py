import argparse
from pathlib import Path

import numpy as np
import torch

from uvnet_retrieval.build_index import load_index, _as_numpy
from uvnet_retrieval.uvnet_wrapper import UVNetEncoder


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_graph(path):
    from dgl.data.utils import load_graphs

    graphs = load_graphs(str(path))[0]
    if not graphs:
        raise ValueError(f"No graphs found in {path}")
    return graphs[0]


def load_checkpoint(model, checkpoint, device):
    if checkpoint is None:
        return
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "student" in payload:
        state = payload["student"]
    else:
        state = payload
    model.load_state_dict(state, strict=False)


def search_index(index_dir, query_emb, top_k=5):
    bundle = load_index(index_dir)
    query = _as_numpy(query_emb).astype("float32")
    if query.ndim == 1:
        query = query[None, :]

    if bundle.backend == "faiss":
        dist, idx = bundle.index.search(query, top_k)
        return idx[0].tolist(), dist[0].tolist()
    if bundle.backend == "sklearn":
        dist, idx = bundle.index.kneighbors(query, n_neighbors=top_k)
        return idx[0].tolist(), dist[0].tolist()
    dist, idx = bundle.index.search(query[0], top_k)
    return idx.tolist(), dist.tolist()


def embed_query(query_bin, checkpoint, device="auto"):
    device = resolve_device(device)
    model = UVNetEncoder().to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    graph = load_graph(query_bin).to(device)
    with torch.no_grad():
        _, graph_emb = model(graph)
    return graph_emb.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser("Search index with a query graph")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--query_bin", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    query = embed_query(args.query_bin, args.checkpoint, device=args.device)
    idx, dist = search_index(args.index_dir, query, top_k=args.top_k)

    bundle = load_index(args.index_dir)
    files = bundle.files or []
    for rank, (i, d) in enumerate(zip(idx, dist), start=1):
        path = files[i] if i < len(files) else f"index:{i}"
        print(f"{rank}\t{d:.6f}\t{path}")

if __name__ == "__main__":
    main()
