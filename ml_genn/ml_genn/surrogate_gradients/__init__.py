from .surrogate_gradient import SurrogateGradient
from .boxcar import Boxcar
from .triangle import Triangle

from ..utils.module import get_module_classes

default_surrogate_gradients = get_module_classes(globals(), SurrogateGradient)

__all = ["Boxcar", "Triangle", "SurrogateGradient", "default_surrogate_gradients"]
