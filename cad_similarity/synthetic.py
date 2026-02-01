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


def sample_surface(mesh, n_points=1000, seed=0):
    np.random.seed(seed)
    pts, faces = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[faces]
    return pts, normals


def sample_part(part, n_base=1000, n_teeth=500, seed=0):
    base_pts, base_normals = sample_surface(part.base_mesh, n_points=n_base, seed=seed)
    teeth_pts, teeth_normals = sample_surface(part.teeth_mesh, n_points=n_teeth, seed=seed + 1)

    pts = np.vstack([base_pts, teeth_pts])
    normals = np.vstack([base_normals, teeth_normals])
    labels = np.hstack([np.zeros(n_base, dtype=int), np.ones(n_teeth, dtype=int)])
    return pts, normals, labels
