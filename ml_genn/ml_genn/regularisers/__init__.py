"""Regularisers are used to control neuron dynamics during training"""
from .regulariser import Regulariser
from .spike_count import SpikeCount

from ..utils.module import get_module_classes

default_regularisers = get_module_classes(globals(), Regulariser)

__all__ = ["Regulariser", "SpikeCount",
           "default_regularisers"]
