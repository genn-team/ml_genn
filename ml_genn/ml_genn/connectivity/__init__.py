"""Connectivity objects are used to describe different sorts of 
connectivity which can be used by :class:`ml_genn.Connection` objects"""
from .avg_pool_2d import AvgPool2D
from .avg_pool_conv_2d import AvgPoolConv2D
from .avg_pool_dense_2d import AvgPoolDense2D
from .connectivity import Connectivity
from .conv_2d import Conv2D
from .conv_2d_transpose import Conv2DTranspose
from .dense import Dense
from .fixed_probability import FixedProbability
from .one_to_one import OneToOne

from ..utils.module import get_module_classes

default_connectivity = get_module_classes(globals(), Connectivity)

__all__ = ["AvgPool2D", "AvgPoolConv2D", "AvgPoolDense2D", "Connectivity",
           "Conv2D", "Conv2DTranspose", "Dense", "FixedProbability",
           "OneToOne", "default_connectivity"]
