import argparse
import json
import random
import time
from pathlib import Path

import torch

from uvnet_retrieval.geometry_targets import compute_geometry_targets
from uvnet_retrieval.heads import GeometryHead
from uvnet_retrieval.losses import masked_mse
from uvnet_retrieval.masking import apply_node_mask, batch_random_mask
from uvnet_retrieval.uvnet_wrapper import UVNetEncoder
from uvnet_retrieval.data.abc_dataset import ABCDataset


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sample_indices(indices, max_items=None, seed=0):
    indices = list(indices)
    if max_items is None or max_items >= len(indices):
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[: max(0, int(max_items))]


def load_graph(path):
    from dgl.data.utils import load_graphs

    graphs = load_graphs(str(path))[0]
    if not graphs:
        raise ValueError(f"No graphs found in {path}")
    return graphs[0]


def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)


def compute_losses(
    graph,
    mask,
    student=None,
    teacher=None,
    geom_head=None,
    device="cpu",
    geom_weight=0.1,
):
    if student is None:
        student = UVNetEncoder().to(device)
    if teacher is None:
        teacher = UVNetEncoder().to(device)
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

    face_feat = graph.ndata["x"].to(device)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device)
    masked_feat = apply_node_mask(face_feat.clone(), mask)
    graph.ndata["x"] = masked_feat
    student_tokens, _ = student(graph)
    with torch.no_grad():
        graph.ndata["x"] = face_feat
        teacher_tokens, _ = teacher(graph)

    loss_c = masked_mse(student_tokens, teacher_tokens, mask)
    if geom_head is None:
        geom_head = GeometryHead(student_tokens.size(-1)).to(device)
    if mask.any():
        geom_pred = geom_head(student_tokens[mask])
        geom_target = compute_geometry_targets(face_feat)[mask]
        loss_a = torch.nn.functional.mse_loss(geom_pred, geom_target)
    else:
        loss_a = torch.tensor(0.0, device=device)
    graph.ndata["x"] = face_feat
    return loss_c + geom_weight * loss_a


def run_epoch(
    files,
    student,
    teacher,
    geom_head,
    optimizer,
    device="cpu",
    mask_ratio=0.6,
    geom_weight=0.1,
    ema_decay=0.99,
    seed=0,
    train=True,
    batch_size=2,
):
    import dgl

    if train:
        student.train()
        geom_head.train()
    else:
        student.eval()
        geom_head.eval()
    teacher.eval()

    total_loss = 0.0
    total_items = 0
    for batch_idx, batch_files in enumerate(iter_batches(files, batch_size=batch_size)):
        graphs = [load_graph(p) for p in batch_files]
        batch_sizes = [g.num_nodes() for g in graphs]
        graph = dgl.batch(graphs).to(device)
        mask = batch_random_mask(
            batch_sizes,
            mask_ratio=mask_ratio,
            seed=seed + batch_idx,
        )

        loss = compute_losses(
            graph,
            mask,
            student=student,
            teacher=teacher,
            geom_head=geom_head,
            device=device,
            geom_weight=geom_weight,
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(teacher, student, ema_decay)

        total_loss += float(loss.item()) * len(batch_files)
        total_items += len(batch_files)

    if total_items == 0:
        return 0.0, 0
    return total_loss / total_items, total_items


def save_checkpoint(path, student, teacher, geom_head, optimizer, epoch, best_loss):
    payload = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "geom_head": geom_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser("Train UV-Net MAE on ABC graphs")
    parser.add_argument("--graph_root", required=True, help="Root folder of .bin graphs")
    parser.add_argument("--runs_root", required=True, help="Output directory for checkpoints/logs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--geom_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    train_ds = ABCDataset(
        root=args.graph_root,
        graph_root=args.graph_root,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    val_ds = ABCDataset(
        root=args.graph_root,
        graph_root=args.graph_root,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_files = train_ds.items
    val_files = val_ds.items

    student = UVNetEncoder().to(device)
    teacher = UVNetEncoder().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    geom_head = GeometryHead(64).to(device)

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(geom_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    log_path = runs_root / "train_log.jsonl"
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_seed = args.seed + epoch
        sample_idx = sample_indices(
            range(len(train_files)),
            max_items=args.max_train_samples,
            seed=epoch_seed,
        )
        sampled_files = [train_files[i] for i in sample_idx]

        t0 = time.time()
        train_loss, train_count = run_epoch(
            sampled_files,
            student,
            teacher,
            geom_head,
            optimizer,
            device=device,
            mask_ratio=args.mask_ratio,
            geom_weight=args.geom_weight,
            ema_decay=args.ema_decay,
            seed=epoch_seed,
            train=True,
            batch_size=args.batch_size,
        )
        val_loss, val_count = run_epoch(
            val_files,
            student,
            teacher,
            geom_head,
            optimizer,
            device=device,
            mask_ratio=args.mask_ratio,
            geom_weight=args.geom_weight,
            ema_decay=args.ema_decay,
            seed=epoch_seed + 1000,
            train=False,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_samples": train_count,
            "val_samples": val_count,
            "seconds": elapsed,
            "device": device,
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        save_checkpoint(runs_root / "checkpoint_last.pt", student, teacher, geom_head, optimizer, epoch, best_loss)
        if val_count > 0 and val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(runs_root / "checkpoint_best.pt", student, teacher, geom_head, optimizer, epoch, best_loss)


if __name__ == "__main__":
    main()
