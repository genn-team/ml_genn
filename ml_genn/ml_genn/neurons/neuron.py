from ..readouts import Readout

from ..utils.module import get_object

from ..readouts import default_readouts


class Neuron:
    def __init__(self, softmax=False, readout=None, **kwargs):
        super(Neuron, self).__init__(**kwargs)
        self.softmax = softmax
        self.readout = get_object(readout, Readout, "Readout",
                                  default_readouts)

    def get_readout(self, genn_pop, batch_size: int, shape):
        if self.readout is None:
            raise RuntimeError("Cannot get readout from neuron "
                               "without readout strategy")
        else:
            return self.readout.get_readout(genn_pop, batch_size, shape)
