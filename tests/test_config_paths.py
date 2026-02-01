import yaml

def test_config_has_step_paths():
    with open("configs/uvnet_mae.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg
    assert "step_root" in cfg["data"]
    assert "graph_root" in cfg["data"]
    assert cfg["data"].get("layout") == "abc"
