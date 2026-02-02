import tempfile
from pathlib import Path

from uvnet_retrieval.data.abc_dataset import ABCDataset, split_indices


def test_split_indices_deterministic():
    train_a, val_a = split_indices(10, val_ratio=0.2, seed=123)
    train_b, val_b = split_indices(10, val_ratio=0.2, seed=123)
    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 2


def test_dataset_max_items_and_split():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(10):
            (root / f"{i:04d}.bin").write_bytes(b"x")
        ds_train = ABCDataset(
            root=str(root), split="train", val_ratio=0.2, seed=7, max_items=3
        )
        ds_val = ABCDataset(root=str(root), split="val", val_ratio=0.2, seed=7)
        assert len(ds_train) == 3
        assert len(ds_val) == 2
