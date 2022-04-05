from ..outputs import Output

from ..utils.module import get_object

from ..outputs import default_outputs


class Neuron:
    def __init__(self, output=None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.output = get_object(output, Output, "Output", default_outputs)
    
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
