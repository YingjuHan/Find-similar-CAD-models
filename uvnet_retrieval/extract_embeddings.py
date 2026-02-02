import argparse
import random
from pathlib import Path

import torch

from uvnet_retrieval.data.abc_dataset import discover_graph_files
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


def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_checkpoint(model, checkpoint, device):
    if checkpoint is None:
        return
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "student" in payload:
        state = payload["student"]
    else:
        state = payload
    model.load_state_dict(state, strict=False)


def sample_files(files, max_items=None, seed=0):
    files = list(files)
    if max_items is None or max_items >= len(files):
        return files
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[: max(0, int(max_items))]


def extract_embeddings(
    graph_root,
    checkpoint,
    out_path,
    batch_size=1,
    device="auto",
    max_items=None,
    seed=0,
):
    import dgl

    device = resolve_device(device)
    files = discover_graph_files(Path(graph_root))
    files = sample_files(files, max_items=max_items, seed=seed)
    if not files:
        raise ValueError("No graph files found")

    model = UVNetEncoder().to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    embeddings = []
    file_list = []
    with torch.no_grad():
        for batch_files in iter_batches(files, batch_size=batch_size):
            graphs = [load_graph(p) for p in batch_files]
            graph = dgl.batch(graphs).to(device)
            _, graph_emb = model(graph)
            embeddings.append(graph_emb.cpu())
            file_list.extend([str(p) for p in batch_files])

    emb = torch.cat(embeddings, dim=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": emb, "files": file_list}, str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser("Extract UV-Net embeddings")
    parser.add_argument("--graph_root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    extract_embeddings(
        graph_root=args.graph_root,
        checkpoint=args.checkpoint,
        out_path=args.out_path,
        batch_size=args.batch_size,
        device=args.device,
        max_items=args.max_items,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
