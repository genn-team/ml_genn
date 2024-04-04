"""Metrics are used for calculating the performance of models
based on some 'true' values and the output of a model obtained 
using a :class:`ml_genn.readouts.Readout`"""
from .mean_square_error import MeanSquareError
from .metric import Metric
from .sparse_categorical_accuracy import SparseCategoricalAccuracy

from ..utils.module import get_module_classes

default_metrics = get_module_classes(globals(), Metric)

__all__ = ["MeanSquareError", "Metric", "SparseCategoricalAccuracy",
           "default_metrics"]
