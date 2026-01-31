import json
from pathlib import Path


def test_notebook_exists_and_valid():
    path = Path("notebooks/cad_similarity_reproduction.ipynb")
    assert path.exists()
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat", 0) >= 4
