"""Connectivity optimisers are used to compute optimise sparse
connectivity during training, typically based on gradient information."""
from .deep_r import DeepR
from .connectivity_optimiser import ConnectivityOptimiser

from ..utils.module import get_module_classes

default_connectivity_optimisers = get_module_classes(globals(),
                                                     ConnectivityOptimiser)

__all__ = ["ConnectivityOptimiser", "DeepR",
           "default_connectivity_optimisers"]
