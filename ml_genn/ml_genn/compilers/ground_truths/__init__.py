"""Prediction classes provide prediction data to loss functions 
and metrics which require device computation"""
from .example_label import ExampleLabel
from .example_value import ExampleValue
from .prediction import Prediction
from .timestep_value import TimestepValue
from ..utils.module import get_module_classes

default_predictions = get_module_classes(globals(), Prediction)

__all__ = ["ExampleLabel", "ExampleValue", "Prediction", 
           "TimestepValue", "default_losses"]
