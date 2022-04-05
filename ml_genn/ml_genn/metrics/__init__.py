from .metric import Metric
from .sparse_categorical_accuracy import SparseCategoricalAccuracy

from ..utils.module import get_module_classes

default_metrics = get_module_classes(globals(), Metric)
