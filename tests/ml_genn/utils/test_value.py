from ml_genn.utils.value import ValueDescriptor

from pytest import approx, raises
from ml_genn.utils.value import get_genn_var_name, get_values

class Model:
    x = ValueDescriptor("X")
    y = ValueDescriptor("Y")
    z = ValueDescriptor(("Z", lambda val, dt: val * dt))

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def test_invalid_values():
    pass

def test_get_genn_name():
    x = Model(1.0, 2.0, 3.0)
    
    assert get_genn_var_name(x, "x") == "X"

    with raises(AttributeError):
        get_genn_var_name(x, "n")

def test_get_values():
    x = Model(1.0, 2.0, 3.0)
    
    vars = [("X", "int"), ("Y", "scalar"), ("Z", "scalar")]
    var_vals = get_values(x, vars, 0.1)
    
    assert var_vals["X"] == approx(1.0)
    assert var_vals["Z"] == approx(0.3)
    assert len(var_vals) == 3
