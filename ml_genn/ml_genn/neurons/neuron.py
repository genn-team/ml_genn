from collections import namedtuple

Model = namedtuple("Model", ["model", "param_vals", "var_vals"])

class Neuron:
    def __init__(self, **kwargs):
        super(Neuron, self).__init__(**kwargs)
