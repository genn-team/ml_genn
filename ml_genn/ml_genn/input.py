from . import encoders

from typing import Sequence, Union
from .encoders import Encoder
from .model import Model

from .utils.model import get_module_models, get_model

# Use Keras-style trick to get dictionary containing default encoder models
_encoder_models = get_module_models(encoders, Encoder)

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

class Input:
    def __init__(self, encoder: Encoder, shape: Shape=None, add_to_model=True):
        self.encoder = get_model(encoder, Encoder, "Encoder", _encoder_models)
        self.shape = _get_shape(shape)
        self.outgoing_connections = []

        # Add input to model
        if add_to_model:
            Model.add_input(self)
    
    # **TODO** shape setter which validate shape with neuron parameters etc
