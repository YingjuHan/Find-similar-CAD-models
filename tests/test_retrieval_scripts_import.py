import importlib

def test_retrieval_imports():
    importlib.import_module("uvnet_retrieval.extract_embeddings")
    importlib.import_module("uvnet_retrieval.build_index")
    importlib.import_module("uvnet_retrieval.search")
