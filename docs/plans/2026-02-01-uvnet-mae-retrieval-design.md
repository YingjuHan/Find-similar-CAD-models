# UV-Net MAE Retrieval Design

**Goal:** Learn a CAD embedding for similarity retrieval using ABC data (STEP/STP), with a MAE-style objective where C (teacher-student embedding prediction) is the main loss and A (light geometry) is an auxiliary loss. B (topology/label) is disabled in this iteration.

**Scope:**
- Input format: STEP/STP only (no data modification).
- Dataset layout: ABC official chunked layout (layout=abc).
- Two-stage pipeline: preprocess STEP -> DGL graphs, then train MAE to produce embeddings for ANN search.

---

## Architecture

### Components
- **Preprocess**: `uvnet_retrieval/preprocess_abc.py` wraps UV-Net `solid_to_graph` to convert each STEP directory into a DGL `.bin` graph with node UV-grid features.
- **Dataset**: `uvnet_retrieval/data/abc_dataset.py` discovers `.bin` graphs from `graph_root` and yields DGL graphs.
- **Encoder**: `uvnet_retrieval/uvnet_wrapper.py` wraps official UV-Net encoders to return per-face tokens and a pooled graph embedding.
- **Training**: `uvnet_retrieval/train_mae.py` implements masking + student/teacher (EMA) losses.
- **Retrieval**: `extract_embeddings.py`, `build_index.py`, `search.py` provide embedding extraction and ANN search.

### Losses
- **C (primary)**: masked-face embedding regression to teacher embeddings (MSE or cosine on normalized tokens).
- **A (aux)**: light geometry regression on masked-face tokens. Targets are computed from existing UV-Net face features (no external labels).
- **B**: disabled (no topology/label supervision in this iteration).

---

## Data Flow

1. **Preprocess**: walk ABC STEP folders (layout=abc), call UV-Net `solid_to_graph`, write `.bin` DGL graphs to `graph_root`.
2. **Load**: dataset yields DGL graphs; dataloader batches graphs.
3. **Mask**: sample per-graph masks by `mask_ratio`; replace masked nodes with a learnable mask token.
4. **Encode**:
   - **Student** runs on masked graph.
   - **Teacher** (EMA, stop-grad) runs on unmasked graph to generate target embeddings.
5. **Loss**:
   - **C**: student masked-face tokens vs teacher tokens.
   - **A**: student masked-face tokens -> geometry head -> light geometry target.

---

## Light Geometry Target (A2)

Computed directly from UV-Net node features:
- **Normal angle distribution**: statistics over UV-grid normals (e.g., mean/variance of pairwise angle or neighbor angle histogram summary).
- **Curvature proxy**: derived from UV-Net surface parameters or UV-grid derivatives if present.

The target is a compact fixed-size vector (e.g., 6-12 dims) per face. No external labels or dataset changes required.

---

## Retrieval Output

- **Embedding**: pooled graph embedding per CAD model.
- **Index**: FAISS (optional) or sklearn NearestNeighbors.
- **Query**: return top-k most similar model IDs.

---

## Configurations

- `configs/uvnet_mae.yaml`: GPU training (A4000).
- `configs/uvnet_mae_debug.yaml`: CPU debug (small batches, few steps).

---

## Testing Strategy

- Unit tests for masking, config paths, dataset discovery on empty roots.
- Unit tests for geometry target computation shape and finiteness using synthetic UV grid tensors.
- No dependency on real ABC data for tests.

---

## Risks and Mitigations

- **Teacher bias**: use light A loss to discourage shortcut embeddings.
- **UV parameterization variance**: use statistical geometry targets rather than direct coordinate regression.
- **Dependency weight**: keep UV-Net wrapper minimal; avoid modifying external submodule.
