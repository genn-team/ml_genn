from .. import outputs

from ..outputs import Output

from ..utils.module import get_module_classes, get_object

# Use Keras-style trick to get dictionary containing default output strategies
_output_strategies = get_module_classes(outputs, Output)

class Neuron:
    def __init__(self, output=None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.output = get_object(output, Output, "Output", _output_strategies)
    
    def add_output_logic(self, model, output_var_name=None):
        if self.output is not None:
            return self.output(model, output_var_name)
        else:
            return model
    
    def get_output(self, genn_pop, batch_size, shape):
        if self.output is None:
            raise RuntimeError("Cannot get output from neuron "
                               "without output strategy")
        else:
            return self.output.get_output(genn_pop, batch_size, shape)
