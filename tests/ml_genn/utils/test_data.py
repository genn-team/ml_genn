import numpy as np

from pytest import raises
from ml_genn.utils.data import (get_dataset_size, split_dataset)

dataset = {"input": [np.ones((2, 2)) * i for i in range(10)]}
dataset_multiple = {"input1": [np.ones((2, 2)) * i for i in range(10)],
                    "input2": [np.ones((2, 2)) * -i for i in range(10)]}

def test_invalid_split_dataset():
    with raises(RuntimeError):
        _, _ = split_dataset(dataset, 10.0)

    with raises(RuntimeError):
        _, _ = split_dataset(dataset, -1.0)

def test_split_dataset():
    # Test corner cases of empty left split
    left, right = split_dataset(dataset, 0.0)
    assert np.allclose(left["input"], dataset["input"])
    assert len(right["input"]) == 0

    # Test corner cases of empty right split
    left, right = split_dataset(dataset, 1.0)
    assert len(left["input"]) == 0
    assert np.allclose(right["input"], dataset["input"])
    
    # Test more normal split
    left, right = split_dataset(dataset, 0.1)
    assert np.allclose(left["input"], dataset["input"][:9])
    assert np.allclose(right["input"], dataset["input"][9:])
    

def test_get_dataset_size():
    assert get_dataset_size(dataset) == 10
    assert get_dataset_size(dataset_multiple) == 10
    
    dataset_invalid = {"input1": [np.ones((2, 2)) * i for i in range(10)],
                       "input2": [np.ones((2, 2)) * -i for i in range(5)]}
    assert get_dataset_size(dataset_invalid) is None