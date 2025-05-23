"""Loss functions are used to compute the quantity that a 
model should seek to minimize during training. For supervised
learning tasks, these are based on some labels and the prediction 
obtained from a model using a :class:`ml_genn.readouts.Readout`"""
from .mean_square_error import MeanSquareError
from .loss import Loss
from .per_neuron_mean_square_error import PerNeuronMeanSquareError
from .sparse_categorical_crossentropy import SparseCategoricalCrossentropy

from ..utils.module import get_module_classes

default_losses = get_module_classes(globals(), Loss)

__all__ = ["MeanSquareError", "Loss", "PerNeuronMeanSquareError",
           "SparseCategoricalCrossentropy", "default_losses"]
