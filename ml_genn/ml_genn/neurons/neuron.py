from .. import outputs

from collections import namedtuple
from ..outputs import Output

from ..utils.model import get_module_models, get_model

# Use Keras-style trick to get dictionary containing default output strategies
_output_strategies = get_module_models(outputs, Output)

Model = namedtuple("Model", ["model", "param_vals", "var_vals"])

class Neuron:
    def __init__(self, output=None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.output = get_model(output, Output, "Output", _output_strategies)
    
    def add_output_logic(self, model, output_var_name=None):
        if self.output is not None:
            return self.output(model, output_var_name)
        else:
            return model
    
    def get_output(self, genn_pop, shape):
        if self.output is None:
            raise RuntimeError("Cannot get output from neuron "
                               "without output strategy")
        else:
            return self.output.get_output(genn_pop, shape)
