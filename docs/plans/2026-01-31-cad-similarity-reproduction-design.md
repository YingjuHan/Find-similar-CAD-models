# CAD Similarity Reproduction Design

Date: 2026-01-31

## Goal
Reproduce the case-study workflow from "Comparing CAD part models for geometrical similarity" using Python, with a synthetic dataset of six parts grouped by tooth geometry. The reproduction will follow the paper's two-stage similarity approach: global similarity on full geometry and local similarity on segmented features (teeth), and provide the requested visualizations in a Jupyter Notebook.

## Scope
- Synthetic data only (no external CAD files).
- Six parts, three groups (two parts per group), grouped by tooth type.
- Full pipeline: global similarity -> clustering -> supervised segmentation -> DBSCAN clustering -> local similarity -> overall ranking.
- Visualizations: global similarity heatmap, segmentation + DBSCAN clusters, and ranked retrieval table.

## Non-Goals
- Exact replication of the original CAD and experimental values.
- Heavy CAD kernels (OCC/FreeCAD).
- Advanced texture/feature descriptors beyond the projection method.

## Data Generation (Synthetic Parts)
Use `trimesh` to assemble parts from primitive meshes without external boolean engines:
- Base body: cylinder (or short thick disk).
- Teeth: small meshes placed along the top rim, three groups by tooth type:
  - Group A: rectangular teeth (boxes).
  - Group B: triangular teeth (custom triangular prism mesh).
  - Group C: arc teeth (small cylinders).
- For each group, create two parts with slight parameter perturbations:
  - Tooth count: e.g., 10-14
  - Tooth size jitter: +/- 5-10%
  - Placement radius jitter: +/- 2-5%
- Labeling: points sampled from tooth sub-meshes are labeled as tooth=1, base=0.

## Sampling and Normalization
- Sample points from each sub-mesh surface (teeth sampled slightly denser).
- Compute surface normals for sampled points.
- Normalize per part: center to origin, scale to fit in unit sphere.
- Fix a global random seed for reproducibility.

## Projection Method
Sphere grid resolution: 224 x 224.

### Global Projection
- Direction vector: from object center to point position.
- Convert direction to spherical coordinates (theta, phi) and map to pixel indices.
- Accumulate point hits per pixel; normalize by total points.

### Local Projection
- Direction vector: surface normal at point.
- Same spherical mapping and accumulation as global projection.

## Similarity Metric
- Use 2D correlation coefficient on flattened matrices.
- Output in [-1, 1].
- If either matrix is constant, define similarity as 0.

## Global Clustering
- Sort matches by similarity.
- Cluster rule: consecutive similarity values within 0.01 are grouped.
- Output: clusters and similarity matrix heatmap.

## Segmentation (Supervised)
### Features
Per point feature vector (example):
- xyz position
- normal xyz
- radius r = ||x||
- mean kNN distance (local density proxy)

### Models
Train and evaluate two classifiers:
- Decision Tree
- MLP (two hidden layers, modest size)

Report train/test accuracy for both. Use the better of the two for downstream clustering, but keep both results in the notebook.

## Tooth Clustering (DBSCAN)
- Apply DBSCAN to points predicted as tooth=1.
- Epsilon set based on mean kNN distance; min_samples scaled by point count.
- Visualize clusters with distinct colors.

## Local Similarity Ranking
- For each global cluster, compute local similarity by comparing tooth-cluster projections.
- Rank parts within each global cluster based on local similarity.
- Combine global cluster order + local ranking to produce final retrieval order.

## Visualizations
Required outputs in the notebook:
- Global similarity heatmap (parts x parts).
- Segmentation visualization (tooth vs base) and DBSCAN cluster view.
- Ranked retrieval table for at least one query part (like paper Fig. 8).

## Notebook Structure
1. Imports, config, random seed
2. Synthetic part generation
3. Sampling + normalization
4. Projection utilities
5. Global similarity + clustering + heatmap
6. Segmentation training (DT + MLP) + accuracy
7. DBSCAN clustering + 3D visualization
8. Local similarity + ranking table
9. Summary of key parameters and results

## Dependencies
- numpy, scipy
- matplotlib
- scikit-learn
- trimesh

## Success Criteria
- Same-group parts appear in the top retrieval results for each query.
- Visualizations A/C/D are produced and match the expected qualitative behavior.
- Notebook runs end-to-end without manual intervention.
