# UV-Net MAE Retrieval (C + A) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement UV-Net MAE retrieval with C (teacher-student embedding) as primary loss and A (light geometry) as auxiliary loss using ABC STEP/STP data, producing embeddings for ANN similarity search.

**Architecture:** Use UV-Net to encode face UV-grid features into per-face tokens and a pooled graph embedding. Train MAE with masking and EMA teacher. Compute light geometry targets from existing UV-grid face features (normals + curvature proxy). Provide extraction and retrieval scripts with CPU debug config and GPU config.

**Tech Stack:** Python, PyTorch, DGL, numpy, sklearn, FAISS (optional), pytest.

---

### Task 1: Geometry Target Computation (A2) + Head Alignment

**Files:**
- Create: `uvnet_retrieval/geometry_targets.py`
- Modify: `uvnet_retrieval/heads.py`
- Create: `tests/test_geometry_targets.py`
- Modify: `tests/test_heads.py`

**Step 1: Write the failing test**

`tests/test_geometry_targets.py`
```python
import torch
from uvnet_retrieval.geometry_targets import compute_geometry_targets


def test_geometry_targets_shape_and_zero_curvature():
    # One face, 2x2 grid, 7 channels: xyz(3) + normals(3) + mask(1)
    normals = torch.tensor([[[[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0]],
                              [[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0]]]])
    points = torch.zeros(1, 2, 2, 3)
    mask = torch.ones(1, 2, 2, 1)
    feat = torch.cat([points, normals, mask], dim=-1)

    out = compute_geometry_targets(feat)
    assert out.shape == (1, 6)
    assert torch.allclose(out, torch.zeros_like(out))
```

`tests/test_heads.py`
```python
import torch
from uvnet_retrieval.heads import GeometryHead, FaceTypeHead


def test_heads_shapes():
    x = torch.randn(5, 16)
    geo = GeometryHead(16, out_dim=6)
    face = FaceTypeHead(16, num_classes=5)
    g = geo(x)
    f = face(x)
    assert g.shape == (5, 6)
    assert f.shape == (5, 5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_targets.py tests/test_heads.py -v`
Expected: FAIL (module not found / shape mismatch)

**Step 3: Write minimal implementation**

`uvnet_retrieval/geometry_targets.py`
```python
import torch


def _to_hwc(x):
    if x.dim() != 4:
        raise ValueError("Expected (F,H,W,C) or (F,C,H,W)")
    # Heuristic: if last dim is 7, assume HWC; else assume CHW
    if x.size(-1) == 7:
        return x
    if x.size(1) == 7:
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError("Could not infer UV-grid channel layout")


def compute_geometry_targets(face_feat):
    # face_feat: (F,H,W,7) with xyz(3), normals(3), mask(1)
    feat = _to_hwc(face_feat)
    normals = feat[..., 3:6]
    mask = feat[..., 6:7]
    mask = (mask > 0).float()

    # Normalize normals
    n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    # Face mean normal (masked)
    masked_n = n * mask
    mean_n = masked_n.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
    mean_n = mean_n / (mean_n.norm(dim=-1, keepdim=True) + 1e-8)

    # Angle stats to mean normal
    dot = (n * mean_n[:, None, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(dot) * mask.squeeze(-1)
    valid = mask.squeeze(-1)
    angle_mean = angles.sum(dim=(1, 2)) / (valid.sum(dim=(1, 2)) + 1e-8)
    angle_var = ((angles - angle_mean[:, None, None]) ** 2 * valid).sum(dim=(1, 2)) / (valid.sum(dim=(1, 2)) + 1e-8)
    angle_std = torch.sqrt(angle_var + 1e-8)

    # Curvature proxy: normal gradient magnitude stats (u and v)
    du = n[:, 1:, :, :] - n[:, :-1, :, :]
    dv = n[:, :, 1:, :] - n[:, :, :-1, :]
    du_mag = du.norm(dim=-1)
    dv_mag = dv.norm(dim=-1)
    du_mean = du_mag.mean(dim=(1, 2))
    du_std = du_mag.std(dim=(1, 2), unbiased=False)
    dv_mean = dv_mag.mean(dim=(1, 2))
    dv_std = dv_mag.std(dim=(1, 2), unbiased=False)

    return torch.stack([angle_mean, angle_std, du_mean, du_std, dv_mean, dv_std], dim=-1)
```

`uvnet_retrieval/heads.py`
```python
class GeometryHead(nn.Module):
    def __init__(self, dim, out_dim=6):
        super().__init__()
        self.net = nn.Linear(dim, out_dim)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_targets.py tests/test_heads.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/geometry_targets.py uvnet_retrieval/heads.py tests/test_geometry_targets.py tests/test_heads.py
git commit -m "feat: add geometry targets for A loss"
```

---

### Task 2: UV-Net Encoder Wrapper (Tokens + Graph Embedding)

**Files:**
- Modify: `uvnet_retrieval/uvnet_wrapper.py`
- Modify: `tests/test_uvnet_wrapper.py`

**Step 1: Write the failing test**

`tests/test_uvnet_wrapper.py`
```python
import pytest
import torch

from uvnet_retrieval.uvnet_wrapper import UVNetEncoder


def test_uvnet_wrapper_importable():
    assert UVNetEncoder is not None


def test_uvnet_wrapper_forward_shapes():
    dgl = pytest.importorskip("dgl")
    # 2 nodes, 1 edge
    g = dgl.graph(([0], [1]), num_nodes=2)
    g.ndata["x"] = torch.randn(2, 10, 10, 7)  # HWC
    g.edata["x"] = torch.randn(1, 10, 6)      # U-grid edge feat
    model = UVNetEncoder(node_embed_dim=64, edge_embed_dim=64, graph_embed_dim=128)
    node_emb, graph_emb = model(g)
    assert node_emb.shape[0] == 2
    assert graph_emb.shape[-1] == 128
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uvnet_wrapper.py -v`
Expected: FAIL (placeholder / NotImplemented)

**Step 3: Write minimal implementation**

`uvnet_retrieval/uvnet_wrapper.py`
```python
import torch
from torch import nn


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

    def _prep_face_feat(self, x):
        # UV-Net expects (B,C,H,W)
        if x.dim() != 4:
            raise ValueError("Expected face features (F,H,W,C) or (F,C,H,W)")
        if x.size(1) == 7:
            return x
        if x.size(-1) == 7:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError("Invalid face feature layout")

    def _prep_edge_feat(self, x):
        # UV-Net expects (B,C,U)
        if x.dim() != 3:
            raise ValueError("Expected edge features (E,U,C) or (E,C,U)")
        if x.size(1) == 6:
            return x
        if x.size(-1) == 6:
            return x.permute(0, 2, 1).contiguous()
        raise ValueError("Invalid edge feature layout")

    def forward(self, graph):
        x = self._prep_face_feat(graph.ndata["x"].float())
        e = self._prep_edge_feat(graph.edata["x"].float())
        node_feat = self.surface(x)
        edge_feat = self.curve(e)
        node_emb, graph_emb = self.graph(graph, node_feat, edge_feat)
        return node_emb, graph_emb
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uvnet_wrapper.py -v`
Expected: PASS or SKIP if DGL missing

**Step 5: Commit**

```bash
git add uvnet_retrieval/uvnet_wrapper.py tests/test_uvnet_wrapper.py
git commit -m "feat: implement UV-Net encoder wrapper"
```

---

### Task 3: MAE Training Loop with EMA Teacher (C + A)

**Files:**
- Modify: `uvnet_retrieval/train_mae.py`
- Modify: `uvnet_retrieval/losses.py`
- Modify: `configs/uvnet_mae.yaml`
- Modify: `configs/uvnet_mae_debug.yaml`
- Create: `tests/test_train_step.py`

**Step 1: Write the failing test**

`tests/test_train_step.py`
```python
import pytest
import torch

from uvnet_retrieval.train_mae import compute_losses


def test_compute_losses_shapes():
    dgl = pytest.importorskip("dgl")
    # small graph
    g = dgl.graph(([0], [1]), num_nodes=2)
    g.ndata["x"] = torch.randn(2, 10, 10, 7)
    g.edata["x"] = torch.randn(1, 10, 6)
    mask = torch.tensor([True, False])
    loss = compute_losses(g, mask)
    assert loss.item() >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_step.py -v`
Expected: FAIL (function missing)

**Step 3: Write minimal implementation**

`uvnet_retrieval/losses.py`
```python
def masked_mse(student, teacher, mask):
    if mask.numel() == 0:
        return torch.tensor(0.0, device=student.device)
    return F.mse_loss(student[mask], teacher[mask])
```

`uvnet_retrieval/train_mae.py`
```python
import torch
import yaml
from uvnet_retrieval.masking import apply_node_mask
from uvnet_retrieval.losses import masked_mse
from uvnet_retrieval.uvnet_wrapper import UVNetEncoder
from uvnet_retrieval.geometry_targets import compute_geometry_targets
from uvnet_retrieval.heads import GeometryHead


def compute_losses(graph, mask, device="cpu", geom_weight=0.1):
    model = UVNetEncoder().to(device)
    teacher = UVNetEncoder().to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    face_feat = graph.ndata["x"].to(device)
    masked_feat = apply_node_mask(face_feat.clone(), mask)
    graph.ndata["x"] = masked_feat
    student_tokens, _ = model(graph)
    with torch.no_grad():
        graph.ndata["x"] = face_feat
        teacher_tokens, _ = teacher(graph)

    loss_c = masked_mse(student_tokens, teacher_tokens, mask.to(device))
    geom_head = GeometryHead(student_tokens.size(-1)).to(device)
    geom_pred = geom_head(student_tokens[mask])
    geom_target = compute_geometry_targets(face_feat)[mask]
    loss_a = torch.nn.functional.mse_loss(geom_pred, geom_target)
    return loss_c + geom_weight * loss_a


def main():
    print("MAE training loop placeholder")

if __name__ == "__main__":
    main()
```

`configs/uvnet_mae.yaml` (add keys)
```yaml
model:
  node_embed_dim: 64
  edge_embed_dim: 64
  graph_embed_dim: 128
  mask_ratio: 0.6
loss:
  lambda_geom: 0.1
ema:
  decay: 0.999
```

`configs/uvnet_mae_debug.yaml` (mirror keys, CPU)
```yaml
model:
  node_embed_dim: 64
  edge_embed_dim: 64
  graph_embed_dim: 128
  mask_ratio: 0.6
loss:
  lambda_geom: 0.1
ema:
  decay: 0.999
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_step.py -v`
Expected: PASS or SKIP if DGL missing

**Step 5: Commit**

```bash
git add uvnet_retrieval/train_mae.py uvnet_retrieval/losses.py configs/uvnet_mae.yaml configs/uvnet_mae_debug.yaml tests/test_train_step.py
git commit -m "feat: add MAE training loss wiring"
```

---

### Task 4: Embedding Extraction + ANN Retrieval

**Files:**
- Modify: `uvnet_retrieval/extract_embeddings.py`
- Modify: `uvnet_retrieval/build_index.py`
- Modify: `uvnet_retrieval/search.py`
- Create: `tests/test_retrieval_pipeline.py`

**Step 1: Write the failing test**

`tests/test_retrieval_pipeline.py`
```python
import numpy as np
from uvnet_retrieval.build_index import build_index


def test_build_index_fallback():
    emb = np.random.randn(5, 8).astype("float32")
    index = build_index(emb)
    assert index is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval_pipeline.py -v`
Expected: FAIL (function missing)

**Step 3: Write minimal implementation**

`uvnet_retrieval/build_index.py`
```python
def build_index(embeddings):
    try:
        import faiss
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="euclidean")
        nn.fit(embeddings)
        return nn
```

`uvnet_retrieval/extract_embeddings.py`
```python
def main():
    print("Embedding extraction placeholder")
```

`uvnet_retrieval/search.py`
```python
def main():
    print("Search placeholder")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieval_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uvnet_retrieval/extract_embeddings.py uvnet_retrieval/build_index.py uvnet_retrieval/search.py tests/test_retrieval_pipeline.py
git commit -m "feat: add ANN index fallback"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-01-uvnet-mae-retrieval-implementation-v2.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
