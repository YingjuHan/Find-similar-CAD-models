# CAD Similarity Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a full, runnable Jupyter Notebook that reproduces the paper's case-study workflow using synthetic CAD-like parts, including global similarity, supervised segmentation, DBSCAN clustering, local similarity, and required visualizations.

**Architecture:** Core logic lives in a small `cad_similarity/` Python package (projection, similarity, synthetic data, segmentation, clustering, local similarity). The notebook `notebooks/cad_similarity_reproduction.ipynb` orchestrates the pipeline and visualizes results. Tests in `tests/` drive TDD for each module.

**Tech Stack:** Python, numpy, scipy, scikit-learn, matplotlib, trimesh, pytest

---

### Task 1: Projection Utilities + Minimal Package Skeleton

**Files:**
- Create: `requirements.txt`
- Create: `cad_similarity/__init__.py`
- Create: `cad_similarity/projection.py`
- Create: `tests/conftest.py`
- Create: `tests/test_projection.py`

**Step 1: Write the failing test**

`tests/test_projection.py`
```python
import numpy as np
from cad_similarity.projection import spherical_projection

def test_spherical_projection_shape_and_sum():
    dirs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    mat = spherical_projection(dirs, bins=(10, 10))
    assert mat.shape == (10, 10)
    assert np.isclose(mat.sum(), 1.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_projection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cad_similarity'`

**Step 3: Write minimal implementation**

`requirements.txt`
```
numpy
scipy
scikit-learn
matplotlib
trimesh
pytest
```

`tests/conftest.py`
```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

`cad_similarity/__init__.py`
```python
__all__ = [
    "projection",
]
```

`cad_similarity/projection.py`
```python
import numpy as np


def spherical_projection(directions, bins=(224, 224)):
    directions = np.asarray(directions, dtype=float)
    # Normalize defensively
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms != 0)

    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arctan2(y, x)  # [-pi, pi]
    phi = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]

    h, w = bins
    u = ((theta + np.pi) / (2 * np.pi) * w).astype(int)
    v = ((phi) / np.pi * h).astype(int)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    mat = np.zeros((h, w), dtype=float)
    for ui, vi in zip(u, v):
        mat[vi, ui] += 1.0

    total = mat.sum()
    if total > 0:
        mat /= total
    return mat
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_projection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt cad_similarity/__init__.py cad_similarity/projection.py tests/conftest.py tests/test_projection.py
git commit -m "feat: add spherical projection utility"
```

---

### Task 2: Similarity Metric + Global Clustering

**Files:**
- Create: `cad_similarity/similarity.py`
- Create: `tests/test_similarity.py`

**Step 1: Write the failing test**

`tests/test_similarity.py`
```python
import numpy as np
from cad_similarity.similarity import correlation_similarity, cluster_by_threshold

def test_correlation_similarity_identity():
    a = np.eye(3)
    b = np.eye(3)
    sim = correlation_similarity(a, b)
    assert np.isclose(sim, 1.0)


def test_cluster_by_threshold():
    sims = [1.0, 0.995, 0.98, 0.975]
    clusters = cluster_by_threshold(sims, threshold=0.01)
    assert clusters == [[0, 1], [2, 3]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_similarity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cad_similarity.similarity'`

**Step 3: Write minimal implementation**

`cad_similarity/similarity.py`
```python
import numpy as np


def correlation_similarity(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    c = np.corrcoef(a, b)
    return float(c[0, 1])


def cluster_by_threshold(similarities, threshold=0.01):
    clusters = []
    current = []
    for idx, val in enumerate(similarities):
        if not current:
            current = [idx]
            last_val = val
            continue
        if abs(last_val - val) <= threshold:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
        last_val = val
    if current:
        clusters.append(current)
    return clusters
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_similarity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cad_similarity/similarity.py tests/test_similarity.py
git commit -m "feat: add similarity metrics and clustering"
```

---

### Task 3: Synthetic Parts + Sampling + Labels

**Files:**
- Create: `cad_similarity/synthetic.py`
- Create: `tests/test_synthetic.py`

**Step 1: Write the failing test**

`tests/test_synthetic.py`
```python
import numpy as np
from cad_similarity.synthetic import generate_parts, sample_part

def test_generate_parts_groups():
    parts = generate_parts(seed=0)
    assert len(parts) == 6
    groups = {p.group for p in parts}
    assert groups == {"rect", "tri", "arc"}


def test_sample_part_counts():
    parts = generate_parts(seed=0)
    pts, normals, labels = sample_part(parts[0], n_base=100, n_teeth=50, seed=0)
    assert pts.shape == (150, 3)
    assert normals.shape == (150, 3)
    assert labels.shape == (150,)
    assert labels.sum() == 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cad_similarity.synthetic'`

**Step 3: Write minimal implementation**

`cad_similarity/synthetic.py`
```python
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class Part:
    name: str
    group: str
    base_mesh: trimesh.Trimesh
    teeth_mesh: trimesh.Trimesh

    @property
    def full_mesh(self):
        return trimesh.util.concatenate([self.base_mesh, self.teeth_mesh])


def _triangular_prism(width=0.2, depth=0.3, height=0.2):
    # Define a simple triangular prism mesh
    vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width / 2.0, depth, 0],
        [0, 0, height],
        [width, 0, height],
        [width / 2.0, depth, height],
    ])
    faces = np.array([
        [0, 1, 2],
        [3, 5, 4],
        [0, 3, 1], [1, 3, 4],
        [1, 4, 2], [2, 4, 5],
        [2, 5, 0], [0, 5, 3],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _place_teeth(meshes, radius, count):
    placed = []
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    for i, ang in enumerate(angles):
        m = meshes[i].copy()
        rot = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])
        trans = trimesh.transformations.translation_matrix([radius, 0, 0])
        m.apply_transform(rot @ trans)
        placed.append(m)
    return trimesh.util.concatenate(placed)


def generate_parts(seed=0):
    rng = np.random.default_rng(seed)
    parts = []

    def make_group(group, tooth_mesh_fn):
        for i in range(2):
            base_radius = 1.0 * (1 + rng.uniform(-0.05, 0.05))
            base_height = 0.4 * (1 + rng.uniform(-0.05, 0.05))
            base = trimesh.creation.cylinder(radius=base_radius, height=base_height, sections=64)

            count = int(rng.integers(10, 14))
            tooth_scale = 0.2 * (1 + rng.uniform(-0.1, 0.1))
            tooth_mesh = tooth_mesh_fn(tooth_scale)
            meshes = [tooth_mesh.copy() for _ in range(count)]
            teeth = _place_teeth(meshes, radius=base_radius * 1.02, count=count)

            parts.append(Part(name=f"{group}-{i}", group=group, base_mesh=base, teeth_mesh=teeth))

    make_group("rect", lambda s: trimesh.creation.box(extents=[s, s * 1.2, s]))
    make_group("tri", lambda s: _triangular_prism(width=s, depth=s * 1.2, height=s))
    make_group("arc", lambda s: trimesh.creation.cylinder(radius=s * 0.4, height=s, sections=16))

    return parts


def sample_part(part, n_base=1000, n_teeth=500, seed=0):
    np.random.seed(seed)
    base_pts, base_faces = trimesh.sample.sample_surface(part.base_mesh, n_base)
    base_normals = part.base_mesh.face_normals[base_faces]

    teeth_pts, teeth_faces = trimesh.sample.sample_surface(part.teeth_mesh, n_teeth)
    teeth_normals = part.teeth_mesh.face_normals[teeth_faces]

    pts = np.vstack([base_pts, teeth_pts])
    normals = np.vstack([base_normals, teeth_normals])
    labels = np.hstack([np.zeros(n_base, dtype=int), np.ones(n_teeth, dtype=int)])
    return pts, normals, labels
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cad_similarity/synthetic.py tests/test_synthetic.py
git commit -m "feat: add synthetic part generation and sampling"
```

---

### Task 4: Segmentation Features + Classifiers

**Files:**
- Create: `cad_similarity/segmentation.py`
- Create: `tests/test_segmentation.py`

**Step 1: Write the failing test**

`tests/test_segmentation.py`
```python
import numpy as np
from cad_similarity.segmentation import build_features, train_models

def test_feature_shape_and_train_models():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 3))
    normals = rng.normal(size=(30, 3))
    X = build_features(points, normals, k=3)
    assert X.shape[0] == 30
    y = np.array([0, 1] * 15)
    models = train_models(X, y, seed=0)
    assert "decision_tree" in models
    assert "mlp" in models
    pred = models["decision_tree"].predict(X[:5])
    assert pred.shape == (5,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cad_similarity.segmentation'`

**Step 3: Write minimal implementation**

`cad_similarity/segmentation.py`
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def build_features(points, normals, k=8):
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    r = np.linalg.norm(points, axis=1, keepdims=True)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(points)))
    nbrs.fit(points)
    dists, _ = nbrs.kneighbors(points)
    # skip self-distance (0)
    mean_knn = dists[:, 1:].mean(axis=1, keepdims=True)

    X = np.hstack([points, normals, r, mean_knn])
    return X


def train_models(X, y, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    dt = DecisionTreeClassifier(random_state=seed)
    dt.fit(X, y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 20),
        random_state=seed,
        max_iter=200,
        solver="lbfgs",
    )
    mlp.fit(X, y)

    return {"decision_tree": dt, "mlp": mlp}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_segmentation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cad_similarity/segmentation.py tests/test_segmentation.py
git commit -m "feat: add segmentation features and classifiers"
```

---

### Task 5: DBSCAN Clustering + Local Similarity

**Files:**
- Create: `cad_similarity/clustering.py`
- Create: `cad_similarity/local_similarity.py`
- Create: `tests/test_clustering.py`
- Create: `tests/test_local_similarity.py`

**Step 1: Write the failing tests**

`tests/test_clustering.py`
```python
import numpy as np
from cad_similarity.clustering import dbscan_clusters


def test_dbscan_finds_two_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.02, size=(30, 3))
    b = rng.normal(loc=1.0, scale=0.02, size=(30, 3))
    pts = np.vstack([a, b])
    labels = dbscan_clusters(pts, eps=0.08, min_samples=5)
    found = set(labels)
    found.discard(-1)
    assert len(found) >= 2
```

`tests/test_local_similarity.py`
```python
import numpy as np
from cad_similarity.local_similarity import local_similarity


def test_local_similarity_identical():
    mats_a = [np.eye(4), np.eye(4)]
    mats_b = [np.eye(4), np.eye(4)]
    sim = local_similarity(mats_a, mats_b)
    assert np.isclose(sim, 1.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clustering.py tests/test_local_similarity.py -v`
Expected: FAIL with `ModuleNotFoundError` for the new modules

**Step 3: Write minimal implementation**

`cad_similarity/clustering.py`
```python
import numpy as np
from sklearn.cluster import DBSCAN


def dbscan_clusters(points, eps=0.05, min_samples=10):
    points = np.asarray(points, dtype=float)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(points)
```

`cad_similarity/local_similarity.py`
```python
import numpy as np
from cad_similarity.similarity import correlation_similarity


def local_similarity(cluster_mats_a, cluster_mats_b):
    if not cluster_mats_a or not cluster_mats_b:
        return 0.0
    sims = []
    for a in cluster_mats_a:
        best = max(correlation_similarity(a, b) for b in cluster_mats_b)
        sims.append(best)
    return float(np.mean(sims))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_clustering.py tests/test_local_similarity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cad_similarity/clustering.py cad_similarity/local_similarity.py tests/test_clustering.py tests/test_local_similarity.py
git commit -m "feat: add dbscan clustering and local similarity"
```

---

### Task 6: Notebook Assembly + Visualizations

**Files:**
- Create: `notebooks/cad_similarity_reproduction.ipynb`
- Create: `tests/test_notebook_exists.py`

**Step 1: Write the failing test**

`tests/test_notebook_exists.py`
```python
import json
from pathlib import Path


def test_notebook_exists_and_valid():
    path = Path("notebooks/cad_similarity_reproduction.ipynb")
    assert path.exists()
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat", 0) >= 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook_exists.py -v`
Expected: FAIL with file not found

**Step 3: Write minimal implementation**

Create `notebooks/cad_similarity_reproduction.ipynb` with these sections:
1. Title + paper reference
2. Imports + config
3. Generate synthetic parts (6 parts, 3 groups)
4. Sampling + normalization (per part)
5. Global projection + similarity matrix + clustering
6. Segmentation training (DecisionTree + MLP) + accuracy
7. DBSCAN clustering visualization
8. Local similarity ranking
9. Summary of parameters

Key code skeleton to include in the notebook cells:
```python
from cad_similarity.synthetic import generate_parts, sample_part
from cad_similarity.projection import spherical_projection
from cad_similarity.similarity import correlation_similarity, cluster_by_threshold
from cad_similarity.segmentation import build_features, train_models
from cad_similarity.clustering import dbscan_clusters
from cad_similarity.local_similarity import local_similarity

# ... rest of pipeline ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook_exists.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add notebooks/cad_similarity_reproduction.ipynb tests/test_notebook_exists.py
git commit -m "feat: add CAD similarity reproduction notebook"
```

---

### Task 7: End-to-End Validation

**Files:**
- Modify: `notebooks/cad_similarity_reproduction.ipynb` (as needed)

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

**Step 2: Manual sanity check in notebook**

Run all cells (Jupyter) and confirm:
- Similarity heatmap shows same-group parts highest
- Segmentation plots show teeth vs base separation
- DBSCAN clusters show multiple tooth clusters
- Retrieval table reflects local re-ranking

**Step 3: Commit (if any changes)**

```bash
git add notebooks/cad_similarity_reproduction.ipynb
git commit -m "chore: finalize notebook outputs"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-31-cad-similarity-notebook-implementation.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
