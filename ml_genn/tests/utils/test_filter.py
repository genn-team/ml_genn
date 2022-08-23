import numpy as np

from ml_genn.utils.filter import ExampleFilter

from itertools import chain
from pytest import raises
from ml_genn.utils.filter import get_neuron_filter_mask

test_mask_1d = [True, False, True, False, True]
test_mask_2d = [[True, False, True], [False, False, False]]
test_mask_2d_flat = list(chain.from_iterable(test_mask_2d))

def test_example_none():
    f = ExampleFilter(None)
    assert np.array_equal(f.get_batch_mask(20, 2), [True, True])

def test_example_bool_array():
    f = ExampleFilter(test_mask_1d)

def test_example_int_array():
    f = ExampleFilter([0, 2, 4])
    
    assert np.array_equal(f.get_batch_mask(0, 2), [True, False])

def test_example_other_object_array():
    with raises(RuntimeError):
        f = ExampleFilter(["hello", 2, 4])
    
def test_example_beyond_mask():
    f = ExampleFilter(test_mask_1d)
    assert np.array_equal(f.get_batch_mask(2, 2), [True, False])

def test_neuron_1d_none():
    mask = get_neuron_filter_mask(None, 5)
    assert np.array_equal(mask, [True, True, True, True, True])
    
def test_neuron_1d_slice():
    mask = get_neuron_filter_mask(slice(0, None, 2), 5)
    assert np.array_equal(mask, test_mask_1d)
    
def test_neuron_1d_bool_array():
    mask = get_neuron_filter_mask(test_mask_1d, 5)
    
def test_neuron_1d_bool_array_bad_shape():
    with raises(RuntimeError):
        mask = get_neuron_filter_mask(test_mask_1d, 7)

def test_neuron_1d_int_array():
    mask = get_neuron_filter_mask([0, 2, 4], 5)
    assert np.array_equal(mask, test_mask_1d)

def test_neuron_1d_int_array_out_of_range():
    with raises(IndexError):
        mask = get_neuron_filter_mask([1, 3, 5], 5)

def test_neuron_1d_other_object_array():
    with raises(RuntimeError):
        mask = get_neuron_filter_mask(["hello", 2, 4], 5)

def test_neuron_2d_none():
    mask = get_neuron_filter_mask(None, (2, 3))
    assert np.array_equal(mask, [True, True, True, True, True, True])

def test_neuron_2d_slice():
    mask = get_neuron_filter_mask((slice(0, None, 2), slice(0, None, 2)), (2, 3))
    assert np.array_equal(mask, test_mask_2d_flat)
    
def test_neuron_2d_bool_array():
    mask = get_neuron_filter_mask(test_mask_2d, (2, 3))
    
def test_neuron_2d_bool_array_bad_shape():
    with raises(RuntimeError):
        mask = get_neuron_filter_mask(test_mask_2d, (3, 3))

def test_neuron_2d_int_array():
    mask = get_neuron_filter_mask([[0, 0], [0, 2]], (2, 3))
    assert np.array_equal(mask, test_mask_2d_flat)

def test_neuron_2d_int_array_out_of_range():
    with raises(IndexError):
        mask = get_neuron_filter_mask([[0, 0], [0, 4]], (2, 3))
        
def test_neuron_2d_int_array_excess_dims():
    with raises(IndexError):
        mask = get_neuron_filter_mask([[0, 0], [0, 2], [0, 0]], (2, 3))
