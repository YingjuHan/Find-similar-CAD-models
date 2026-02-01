import argparse
from uvnet_retrieval.preprocess_abc import mirror_output_dir


def test_mirror_output_dir(tmp_path):
    root = tmp_path / "chunks"
    step_dir = root / "0000" / "step"
    out_root = tmp_path / "out"
    step_dir.mkdir(parents=True, exist_ok=True)
    out_dir = mirror_output_dir(step_dir, root, out_root)
    assert out_dir == out_root / "0000" / "step"
