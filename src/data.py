from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CUBOIDS_DIR = ROOT / "cuboids-v2"
OUTPUT_DIR = ROOT / "outputs"


def list_samples(root=CUBOIDS_DIR):
    """Sample folder names, sorted by their leading numeric id."""
    names = [p.name for p in Path(root).iterdir() if p.is_dir()]
    return sorted(names, key=lambda n: int(n.split("_")[0]))


def load_mesh(obj_path):
    """Vertices [N, 3] and 0-based triangles [M, 3] from an .obj file."""
    verts, faces = [], []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([p.split("/")[0] for p in line.split()[1:4]])
    return np.array(verts, dtype=float), np.array(faces, dtype=int) - 1


def load_trajectory(txt_path):
    """Strokes from a trajectory.txt (X;Y;Z;A;B;C;strokeId), each [K, 6]."""
    data = np.loadtxt(txt_path, delimiter=";", skiprows=1)
    ids = data[:, 6].astype(int)
    return [data[ids == s, :6] for s in np.unique(ids)]


def load_sample(index, root=CUBOIDS_DIR):
    """The index-th sample in list_samples order: name, vertices, faces, strokes."""
    name = list_samples(root)[index]
    folder = Path(root) / name
    vertices, faces = load_mesh(folder / f"{name}.obj")
    return {"name": name, "vertices": vertices, "faces": faces,
            "strokes": load_trajectory(folder / "trajectory.txt")}
