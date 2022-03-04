from . import neurons

from typing import Sequence, Union
from . import Model
from .neurons import Neuron

from .utils.model import get_module_models, get_model

# Use Keras-style trick to get dictionary containing default neuron models
_neuron_models = get_module_models(neurons, Neuron)

def _get_shape(shape):
    if shape is None or isinstance(shape, Sequence):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise RuntimeError("Population shapes should either be left "
                           "unspecified with None or specified as an "
                           "integer or a sequence")

class Population:
    def __init__(self, neuron: Neuron, shape: Union[None, int, Sequence[int]]=None):
        
        self.neuron = get_model(neuron, Neuron, "Neuron", _neuron_models)
        self.shape = _get_shape(shape)
        self.incoming_connections = []
        self.outgoing_connections = []

        # Add population to model
        Model.add_population(self)
