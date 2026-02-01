from pathlib import Path

class ABCDataset:
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split
        if not self.root.exists():
            self.items = []
        else:
            self.items = []  # TODO: load real ABC indices later
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        raise NotImplementedError("ABC data loading not wired")
