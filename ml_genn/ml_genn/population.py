from typing import Sequence, Union
from .network import Network
from .neurons import Neuron

from .utils.module import get_object

from .neurons import default_neurons

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
        self.neuron = get_object(neuron, Neuron, "Neuron", default_neurons)
        self.shape = _get_shape(shape)
        self.incoming_connections = []
        self.outgoing_connections = []

        # Add population to model
        if add_to_model:
            Network.add_population(self)
    
    # **TODO** shape setter which validate shape with neuron parameters etc
