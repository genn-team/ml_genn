import numpy as np

from ml_genn.utils.filter import BatchFilter, NeuronFilter

from pytest import raises

test_mask_1d = [True, False, True, False, True]
test_mask_2d = [[True, False, True], [False, False, False]]

def test_batch_bool():
    f = BatchFilter(True)
    assert f[2] == True

def test_batch_bool_array():
    f = BatchFilter(test_mask_1d)

def test_batch_int_array():
    f = BatchFilter([0, 2, 4])
    assert np.array_equal(f._mask, test_mask_1d)

def test_batch_other_object_array():
    with raises(RuntimeError):
        f = BatchFilter(["hello", 2, 4])
    
def test_batch_beyond_mask():
    f = BatchFilter(test_mask_1d)
    assert not f[10]

def test_neuron_1d_bool():
    f = NeuronFilter(True, 100)
    assert f[10] == True
    
def test_neuron_1d_slice():
    f = NeuronFilter(slice(0, None, 2), 5)
    assert np.array_equal(f._mask, test_mask_1d)
    
def test_neuron_1d_bool_array():
    f = NeuronFilter(test_mask_1d, 5)
    
def test_neuron_1d_bool_array_bad_shape():
    with raises(RuntimeError):
        f = NeuronFilter(test_mask_1d, 7)

def test_neuron_1d_int_array():
    f = NeuronFilter([0, 2, 4], 5)
    assert np.array_equal(f._mask, test_mask_1d)

def test_neuron_1d_int_array_out_of_range():
    with raises(IndexError):
        f = NeuronFilter([1, 3, 5], 5)

def test_neuron_1d_other_object_array():
    with raises(RuntimeError):
        f = NeuronFilter(["hello", 2, 4], 5)
        
def test_neuron_2d_bool():
    f = NeuronFilter(True, (2, 3))
    assert f[1, 2] == True

def test_neuron_2d_slice():
    f = NeuronFilter((slice(0, None, 2), slice(0, None, 2)), (2, 3))
    assert np.array_equal(f._mask, test_mask_2d)
    
def test_neuron_2d_bool_array():
    f = NeuronFilter(test_mask_2d, (2, 3))
    
def test_neuron_2d_bool_array_bad_shape():
    with raises(RuntimeError):
        f = NeuronFilter(test_mask_2d, (3, 3))

def test_neuron_2d_int_array():
    f = NeuronFilter([[0, 0], [0, 2]], (2, 3))
    assert np.array_equal(f._mask, test_mask_2d)

def test_neuron_2d_int_array_out_of_range():
    with raises(IndexError):
        f = NeuronFilter([[0, 0], [0, 4]], (2, 3))
        
def test_neuron_2d_int_array_excess_dims():
    with raises(IndexError):
        f = NeuronFilter([[0, 0], [0, 2], [0, 0]], (2, 3))