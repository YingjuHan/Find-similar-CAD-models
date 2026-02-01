from pathlib import Path


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


class ABCDataset:
    def __init__(self, root, split="train", graph_root=None, layout="abc"):
        self.root = Path(root)
        self.split = split
        self.layout = layout
        self.graph_root = Path(graph_root) if graph_root is not None else self.root
        if not self.graph_root.exists():
            self.items = []
        else:
            self.items = discover_graph_files(self.graph_root)

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
