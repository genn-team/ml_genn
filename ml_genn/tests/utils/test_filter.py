import numpy as np

from ml_genn.utils.filter import Filter

from pytest import raises

test_mask = [True, False, True, False, True]

def test_slice():
    f = Filter(slice(0, None, 2), 5)
    assert np.array_equal(f._mask, test_mask)
    
def test_slice_no_count():
    with raises(RuntimeError):
        f = Filter(slice(0, None, 2))

def test_bool_array():
    f = Filter(test_mask)
    
def test_bool_array_bad_count():
    with raises(RuntimeError):
        f = Filter(test_mask, 7)

def test_int_array():
    f = Filter([0, 2, 4])
    assert np.array_equal(f._mask, test_mask)

def test_int_array_negative():
    with raises(RuntimeError):
        f = Filter([-1, 1, 3])

def test_int_array_beyond_count():
    with raises(RuntimeError):
        f = Filter([1, 3, 5], 5)

def test_out_of_range():
    f = Filter(test_mask)
    assert not f[10]

def test_negative():
    f = Filter(test_mask)
    with raises(IndexError):
        m = f[-1]