from pathlib import Path
import random


def _collect_with_patterns(root, patterns):
    files = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    unique = sorted({p.resolve() for p in files if p.is_file()})
    return unique


def discover_step_files(root, layout="abc"):
    root = Path(root)
    if not root.exists():
        return []
    if layout == "abc":
        patterns = ["**/step/*.stp", "**/step/*.step", "**/STEP/*.stp", "**/STEP/*.step"]
    else:
        patterns = ["**/*.stp", "**/*.step"]
    return _collect_with_patterns(root, patterns)


def discover_step_dirs(root, layout="abc"):
    files = discover_step_files(root, layout=layout)
    dirs = sorted({p.parent.resolve() for p in files})
    return dirs


def discover_graph_files(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted({p.resolve() for p in root.glob("**/*.bin") if p.is_file()})


def split_indices(count, val_ratio=0.1, seed=0):
    if count <= 0:
        return [], []
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(count * val_ratio)
    val_count = max(0, min(count, val_count))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return train_idx, val_idx


def select_split(files, split="train", val_ratio=0.1, seed=0, max_items=None):
    if split == "all":
        selected = list(files)
    else:
        train_idx, val_idx = split_indices(len(files), val_ratio=val_ratio, seed=seed)
        if split == "train":
            idx = train_idx
        elif split == "val":
            idx = val_idx
        else:
            raise ValueError(f"Unsupported split: {split}")
        selected = [files[i] for i in idx]
    if max_items is not None:
        return selected[: max(0, int(max_items))]
    return selected


class ABCDataset:
    def __init__(
        self,
        root,
        split="train",
        graph_root=None,
        layout="abc",
        val_ratio=0.1,
        seed=0,
        max_items=None,
    ):
        self.root = Path(root)
        self.split = split
        self.layout = layout
        self.graph_root = Path(graph_root) if graph_root is not None else self.root
        if not self.graph_root.exists():
            self.items = []
        else:
            files = discover_graph_files(self.graph_root)
            self.items = select_split(
                files,
                split=split,
                val_ratio=val_ratio,
                seed=seed,
                max_items=max_items,
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx >= len(self.items):
            raise IndexError(idx)
        try:
            from dgl.data.utils import load_graphs
        except Exception as exc:  # pragma: no cover - depends on optional dgl
            raise RuntimeError("dgl is required to load graphs") from exc
        graphs = load_graphs(str(self.items[idx]))[0]
        return graphs
