from itertools import count
from typing import Optional, Sequence, Union
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
NeuronInitializer = Union[Neuron, str]


class Population:
    _new_id = count()
    
    def __init__(self, neuron: NeuronInitializer, shape: Shape = None,
                 record_spikes: bool = False, name: Optional[str] = None,
                 add_to_model: bool = True):
        self.neuron = get_object(neuron, Neuron, "Neuron", default_neurons)
        self.shape = _get_shape(shape)
        self.incoming_connections = []
        self.outgoing_connections = []
        self.record_spikes = record_spikes

        # Generate unique name if required
        self.name = (f"Pop{next(Population._new_id)}" if name is None
                     else name)

        # Add population to model
        if add_to_model:
            Network.add_population(self)

    # **TODO** shape setter which validate shape with neuron parameters etc

    def __str__(self):
        return self.name
