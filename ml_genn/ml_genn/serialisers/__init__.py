from .serialiser import Serialiser
from .numpy import Numpy

from ..utils.module import get_module_classes

default_serialisers = get_module_classes(globals(), Serialiser)