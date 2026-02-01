import json
from pathlib import Path


def test_notebook_injects_repo_root():
    path = Path("notebooks/cad_similarity_reproduction.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert code_cells, "No code cells found in notebook"
    first_code = "".join(code_cells[0].get("source", []))
    assert "sys.path" in first_code
    assert "cad_similarity" in first_code
