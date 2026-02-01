import importlib

def test_train_script_importable():
    importlib.import_module("uvnet_retrieval.train_mae")
