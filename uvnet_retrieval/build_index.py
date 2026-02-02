import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class IndexBundle:
    backend: str
    index: object
    embeddings: np.ndarray
    files: list


class NumpyIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query, top_k):
        dists = np.linalg.norm(self.embeddings - query[None, :], axis=1)
        idx = np.argsort(dists)[:top_k]
        return dists[idx], idx


def _as_numpy(embeddings):
    if torch.is_tensor(embeddings):
        return embeddings.detach().cpu().numpy().astype("float32")
    return np.asarray(embeddings, dtype="float32")


def build_index(embeddings):
    embeddings = _as_numpy(embeddings)
    try:
        import faiss

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="euclidean")
            nn.fit(embeddings)
            return nn
        except Exception:
            return NumpyIndex(embeddings)


def _detect_backend(index):
    if index.__class__.__module__.startswith("faiss"):
        return "faiss"
    if hasattr(index, "kneighbors"):
        return "sklearn"
    if isinstance(index, NumpyIndex):
        return "numpy"
    return "numpy"


def save_index(index, embeddings, out_dir, files=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _as_numpy(embeddings)
    np.save(out_dir / "embeddings.npy", embeddings)

    files = list(files) if files is not None else []
    if files:
        with open(out_dir / "files.json", "w", encoding="utf-8") as handle:
            json.dump(files, handle)

    backend = _detect_backend(index)
    meta = {"backend": backend, "shape": list(embeddings.shape)}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle)

    if backend == "faiss":
        import faiss

        faiss.write_index(index, str(out_dir / "index.faiss"))
    elif backend == "sklearn":
        with open(out_dir / "index.pkl", "wb") as handle:
            pickle.dump(index, handle)


def load_index(out_dir):
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = meta.get("backend", "numpy")
    else:
        backend = "numpy"

    embeddings = np.load(out_dir / "embeddings.npy")
    files_path = out_dir / "files.json"
    files = json.loads(files_path.read_text(encoding="utf-8")) if files_path.exists() else []

    if backend == "faiss":
        import faiss

        index = faiss.read_index(str(out_dir / "index.faiss"))
    elif backend == "sklearn":
        with open(out_dir / "index.pkl", "rb") as handle:
            index = pickle.load(handle)
    else:
        index = NumpyIndex(embeddings)
        backend = "numpy"

    return IndexBundle(backend=backend, index=index, embeddings=embeddings, files=files)


def _load_embeddings_file(path):
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        payload = torch.load(path, map_location="cpu")
        embeddings = payload.get("embeddings", payload)
        files = payload.get("files", [])
        return embeddings, files
    if path.suffix == ".npy":
        return np.load(path), []
    raise ValueError(f"Unsupported embeddings file: {path}")


def main():
    parser = argparse.ArgumentParser("Build ANN index")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.pt or embeddings.npy")
    parser.add_argument("--out_dir", required=True, help="Directory to write index files")
    args = parser.parse_args()

    embeddings, files = _load_embeddings_file(args.embeddings)
    index = build_index(embeddings)
    save_index(index, embeddings, args.out_dir, files=files)

if __name__ == "__main__":
    main()
