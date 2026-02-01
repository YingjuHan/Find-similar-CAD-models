import importlib

def test_uvnet_wrapper_module_present():
    spec = importlib.util.find_spec("uvnet_retrieval.uvnet_wrapper")
    assert spec is not None
