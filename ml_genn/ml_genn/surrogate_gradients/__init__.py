from .surrogate_gradient import SurrogateGradient
from .eprop import EProp

from ..utils.module import get_module_classes

default_surrogate_gradients = get_module_classes(globals(), SurrogateGradient)

__all = ["EProp", "SurrogateGradient", "default_surrogate_gradients"]
