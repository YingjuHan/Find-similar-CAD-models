import yaml

def test_debug_config_exists():
    with open("configs/uvnet_mae_debug.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["train"]["device"] == "cpu"
