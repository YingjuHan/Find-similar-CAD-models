# UV-Net MAE Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a UV?Net¨Cbased MAE (C + light A + light B) training and retrieval pipeline using the ABC dataset, producing embeddings for CAD similarity search and a reproducible CPU debug path.

**Architecture:** Integrate official UV?Net as a git submodule, wrap its encoder for embedding extraction, implement MAE masking/teacher?student losses plus light geometric and face?type heads, and provide scripts for training, embedding extraction, and ANN indexing. Unit tests cover masking, loss wiring, and dataset interface without requiring real ABC data.

**Tech Stack:** Python, PyTorch (from UV?Net), numpy, scikit-learn, faiss (optional for ANN), pytest

---

### Task 1: Add UV?Net Submodule + Repo Wrapper Skeleton

**Files:**
- Create: `external/uvnet/` (git submodule)
- Create: `uvnet_retrieval/__init__.py`
- Create: `uvnet_retrieval/uvnet_wrapper.py`
- Create: `tests/test_uvnet_wrapper.py`
- Modify: `requirements.txt`

**Step 1: Write the failing test**

`tests/test_uvnet_wrapper.py`
```python
import importlib

def test_uvnet_submodule_present():
    spec = importlib.util.find_spec("uvnet_retrieval")
    assert spec is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_uvnet_wrapper.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

- Add submodule (verify repo URL first):
```bash
git ls-remote https://github.com/AutodeskAILab/UV-Net
mkdir -p external

git submodule add https://github.com/AutodeskAILab/UV-Net external/uvnet
```

`uvnet_retrieval/__init__.py`
```python
__all__ = ["uvnet_wrapper"]
```

`uvnet_retrieval/uvnet_wrapper.py`
```python
# Placeholder wrapper for UV?Net encoder; actual wiring added in later tasks.
class UVNetEncoder:
    def __init__(self):
        raise NotImplementedError("UV?Net wrapper not yet wired")
```

`requirements.txt` (append UV?Net runtime deps once confirmed from submodule):
```
torch
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_uvnet_wrapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add external/uvnet uvnet_retrieval/__init__.py uvnet_retrieval/uvnet_wrapper.py tests/test_uvnet_wrapper.py requirements.txt
git commit -m "feat: add UV-Net submodule and wrapper scaffold"
```

---

### Task 2: Masking + Teacher?Student Loss (C)

**Files:**
- Create: `uvnet_retrieval/masking.py`
- Create: `uvnet_retrieval/losses.py`
- Create: `tests/test_masking.py`
- Create: `tests/test_losses.py`

**Step 1: Write the failing tests**

`tests/test_masking.py`
```python
import numpy as np
from uvnet_retrieval.masking import random_mask

def test_random_mask_ratio():
    mask = random_mask(100, mask_ratio=0.6, seed=0)
    assert mask.sum() == 60
```

`tests/test_losses.py`
```python
import torch
from uvnet_retrieval.losses import embedding_mse

def test_embedding_mse_zero():
    x = torch.zeros(4, 8)
    y = torch.zeros(4, 8)
    assert embedding_mse(x, y).item() == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_masking.py tests/test_losses.py -v`
Expected: FAIL (modules not found)

**Step 3: Write minimal implementation**

`uvnet_retrieval/masking.py`
```python
import numpy as np

def random_mask(n_tokens, mask_ratio=0.6, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_tokens, dtype=bool)
    idx = rng.choice(n_tokens, int(n_tokens * mask_ratio), replace=False)
    mask[idx] = True
    return mask
```

`uvnet_retrieval/losses.py`
```python
import torch
import torch.nn.functional as F


def embedding_mse(student, teacher):
    return F.mse_loss(student, teacher)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_masking.py tests/test_losses.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/masking.py uvnet_retrieval/losses.py tests/test_masking.py tests/test_losses.py
git commit -m "feat: add MAE masking and embedding loss"
```

---

### Task 3: Light Geometry Head (A) + Light Face?Type Head (B)

**Files:**
- Create: `uvnet_retrieval/heads.py`
- Create: `tests/test_heads.py`

**Step 1: Write the failing test**

`tests/test_heads.py`
```python
import torch
from uvnet_retrieval.heads import GeometryHead, FaceTypeHead

def test_heads_shapes():
    x = torch.randn(5, 16)
    geo = GeometryHead(16)
    face = FaceTypeHead(16, num_classes=5)
    g = geo(x)
    f = face(x)
    assert g.shape == (5, 4)  # e.g., curvature/area/normal stats
    assert f.shape == (5, 5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_heads.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

`uvnet_retrieval/heads.py`
```python
import torch
import torch.nn as nn

class GeometryHead(nn.Module):
    def __init__(self, dim, out_dim=4):
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_heads.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/heads.py tests/test_heads.py
git commit -m "feat: add geometry and face-type heads"
```

---

### Task 4: ABC Dataset Interface (Placeholder, No Data Required)

**Files:**
- Create: `uvnet_retrieval/data/abc_dataset.py`
- Create: `uvnet_retrieval/data/transforms.py`
- Create: `tests/test_dataset_interface.py`

**Step 1: Write failing test**

`tests/test_dataset_interface.py`
```python
from uvnet_retrieval.data.abc_dataset import ABCDataset

def test_dataset_interface_empty():
    ds = ABCDataset(root="/nonexistent", split="train")
    assert len(ds) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_interface.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

`uvnet_retrieval/data/abc_dataset.py`
```python
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
```

`uvnet_retrieval/data/transforms.py`
```python
# Placeholder for future UV?Net feature transforms
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_interface.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/data/abc_dataset.py uvnet_retrieval/data/transforms.py tests/test_dataset_interface.py
git commit -m "feat: add ABC dataset interface placeholder"
```

---

### Task 5: MAE Training Script Skeleton

**Files:**
- Create: `uvnet_retrieval/train_mae.py`
- Create: `configs/uvnet_mae.yaml`
- Create: `tests/test_train_script_import.py`

**Step 1: Write failing test**

`tests/test_train_script_import.py`
```python
import importlib

def test_train_script_importable():
    importlib.import_module("uvnet_retrieval.train_mae")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_script_import.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`uvnet_retrieval/train_mae.py`
```python
def main():
    print("MAE training scaffold")

if __name__ == "__main__":
    main()
```

`configs/uvnet_mae.yaml`
```yaml
data:
  root: /path/to/ABC
  split: train
model:
  mask_ratio: 0.6
  embedding_dim: 256
loss:
  lambda_geom: 0.2
  lambda_face: 0.2
train:
  epochs: 100
  batch_size: 8
  device: cuda
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_script_import.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/train_mae.py configs/uvnet_mae.yaml tests/test_train_script_import.py
git commit -m "feat: add MAE training scaffold"
```

---

### Task 6: Embedding Extraction + Retrieval Scripts (Scaffold)

**Files:**
- Create: `uvnet_retrieval/extract_embeddings.py`
- Create: `uvnet_retrieval/build_index.py`
- Create: `uvnet_retrieval/search.py`
- Create: `tests/test_retrieval_scripts_import.py`

**Step 1: Write failing test**

`tests/test_retrieval_scripts_import.py`
```python
import importlib

def test_retrieval_imports():
    importlib.import_module("uvnet_retrieval.extract_embeddings")
    importlib.import_module("uvnet_retrieval.build_index")
    importlib.import_module("uvnet_retrieval.search")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval_scripts_import.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Each script prints a placeholder message (to be filled once UV?Net wiring exists).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval_scripts_import.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/extract_embeddings.py uvnet_retrieval/build_index.py uvnet_retrieval/search.py tests/test_retrieval_scripts_import.py
git commit -m "feat: add retrieval script scaffolds"
```

---

### Task 7: End?to?End CPU Debug Path (No Real Data)

**Files:**
- Modify: `configs/uvnet_mae.yaml`
- Create: `configs/uvnet_mae_debug.yaml`

**Step 1: Write failing test**

`tests/test_debug_config.py`
```python
import yaml

def test_debug_config_exists():
    with open("configs/uvnet_mae_debug.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["train"]["device"] == "cpu"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_debug_config.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`configs/uvnet_mae_debug.yaml`
```yaml
data:
  root: /path/to/ABC
  split: train
model:
  mask_ratio: 0.6
  embedding_dim: 64
loss:
  lambda_geom: 0.2
  lambda_face: 0.2
train:
  epochs: 1
  batch_size: 1
  device: cpu
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_debug_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add configs/uvnet_mae_debug.yaml tests/test_debug_config.py
git commit -m "feat: add CPU debug config"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-01-uvnet-mae-retrieval-implementation.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
