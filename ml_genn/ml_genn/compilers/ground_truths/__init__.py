"""Ground truth classes provide ground truth data to loss functions 
and metrics which require device computation"""
from .example_label import ExampleLabel
from .example_value import ExampleValue
from .ground_truth import GroundTruth
from .timestep_value import TimestepValue
from ml_genn.utils.module import get_module_classes

default_ground_truths = get_module_classes(globals(), GroundTruth)

__all__ = ["ExampleLabel", "ExampleValue", "GroundTruth", 
           "TimestepValue", "default_ground_truths"]
