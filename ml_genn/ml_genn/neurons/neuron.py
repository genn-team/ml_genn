from ..outputs import Output

from ..utils.module import get_object

from ..outputs import default_outputs


class Neuron:
    def __init__(self, softmax=False, output=None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.softmax = softmax
        self.output = get_object(output, Output, "Output", default_outputs)

    def get_output(self, genn_pop, batch_size: int, shape):
        if self.output is None:
            raise RuntimeError("Cannot get output from neuron "
                               "without output strategy")
        else:
            return self.output.get_output(genn_pop, batch_size, shape)
