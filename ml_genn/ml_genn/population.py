from . import neurons

from typing import Sequence, Union
from .network import Network
from .neurons import Neuron

from .utils.module import get_module_classes, get_object

# Use Keras-style trick to get dictionary containing default neuron models
_neuron_models = get_module_classes(neurons, Neuron)

def _get_shape(shape):
    if shape is None or isinstance(shape, Sequence):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise RuntimeError("Population shapes should either be left "
                           "unspecified with None or specified as an "
                           "integer or a sequence")

Shape = Union[None, int, Sequence[int]]

class Population:
    def __init__(self, neuron: Neuron, shape: Shape=None, add_to_model=True):
        self.neuron = get_object(neuron, Neuron, "Neuron", _neuron_models)
        self.shape = _get_shape(shape)
        self.incoming_connections = []
        self.outgoing_connections = []

        # Add population to model
        if add_to_model:
            Network.add_population(self)
    
    # **TODO** shape setter which validate shape with neuron parameters etc