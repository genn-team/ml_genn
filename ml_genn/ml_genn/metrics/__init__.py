"""Metrics are used for calculating the performance of models
based on some labels and the prediction obtained from a model
using a :class:`ml_genn.readouts.Readout`"""
from typing import Union
from .mean_square_error import MeanSquareError
from .metric import Metric
from .sparse_categorical_accuracy import SparseCategoricalAccuracy

from ..utils.module import get_module_classes

MetricType = Union[Metric, str]
MetricsType = Union[dict, MetricType]

default_metrics = get_module_classes(globals(), Metric)

__all__ = ["MeanSquareError", "Metric", "MetricType", "MetricsType",
           "SparseCategoricalAccuracy", "default_metrics"]
