from pathlib import Path
from uvnet_retrieval.data.abc_dataset import discover_step_files, discover_step_dirs


def test_discover_step_files_abc_layout(tmp_path):
    f1 = tmp_path / "chunks" / "0000" / "step" / "a.stp"
    f2 = tmp_path / "chunks" / "0001" / "step" / "b.step"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("")
    f2.write_text("")

    files = discover_step_files(tmp_path, layout="abc")
    assert len(files) == 2


def test_discover_step_dirs(tmp_path):
    f1 = tmp_path / "chunks" / "0000" / "step" / "a.stp"
    f2 = tmp_path / "chunks" / "0000" / "step" / "b.step"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("")
    f2.write_text("")

    dirs = discover_step_dirs(tmp_path, layout="abc")
    assert len(dirs) == 1
    assert Path(dirs[0]).name.lower() == "step"
