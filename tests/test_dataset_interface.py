from uvnet_retrieval.data.abc_dataset import ABCDataset

def test_dataset_interface_empty():
    ds = ABCDataset(root="/nonexistent", split="train")
    assert len(ds) == 0
