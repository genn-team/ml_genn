from .mean_square_error import MeanSquareError
from .loss import Loss
from .sparse_categorical_crossentropy import SparseCategoricalCrossEntropy

from ..utils.module import get_module_classes

default_losses = get_module_classes(globals(), Loss)
