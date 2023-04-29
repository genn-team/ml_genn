from ml_genn import Network

from ml_genn.utils.module import get_module_classes, get_object
from ml_genn.utils.decorators import network_default_params

class Model:
    @network_default_params
    def __init__(self, a = 1.0, b = 0.0, c=None):
        self.a = a
        self.b = b
        self.c = c

default_neurons = get_module_classes(globals(), object)

def test_defaults():
    m = Model()
    assert m.a == 1.0
    assert m.b == 0.0
    assert m.c == None

    m = get_object("model", object, "", default_neurons)
    assert m.a == 1.0
    assert m.b == 0.0
    assert m.c == None

def test_network_defaults():
    net = Network({Model: {"a": 3.0}})
    with net:
        m = Model()
        assert m.a == 3.0
        assert m.b == 0.0
        assert m.c == None

        m = get_object("model", object, "", default_neurons)
        assert m.a == 3.0
        assert m.b == 0.0
        assert m.c == None

def test_overriden_network_defaults():
    net = Network({Model: {"a": 3.0}})
    with net:
        m = Model(2.0)
        assert m.a == 2.0
        assert m.b == 0.0
        assert m.c == None

        m = Model(a=2.0)
        assert m.a == 2.0
        assert m.b == 0.0
        assert m.c == None
