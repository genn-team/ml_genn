from ml_genn.utils.value import ValueDescriptor

from pytest import approx, raises
from ml_genn.utils.value import (get_auto_values, get_genn_var_name,
                                 get_values)

class Model:
    x = ValueDescriptor()
    y = ValueDescriptor()
    z = ValueDescriptor()

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def test_invalid_values():
    pass
   
def test_get_values():
    x = Model(1.0, 2.0, 3.0)
    
    vars = [("x", "int"), ("y", "scalar"), ("z", "scalar")]
    var_vals = get_values(x, vars)
    
    assert var_vals["x"] == approx(1.0)
    assert var_vals["z"] == approx(0.3)
    assert len(var_vals) == 3

def test_get_auto_values():
    x = Model(1.0, 2.0, 3.0)
    
    var_names = ["x", "y"]
    var_vals, param_vals = get_auto_values(x, vars)
    
    assert var_vals["x"] == approx(1.0)
    assert var_vals["y"] == approx(2.0)
    assert len(var_vals) == 2
    
    assert param_vals["z"] == approx(0.3)
    assert len(param_vals) == 1
