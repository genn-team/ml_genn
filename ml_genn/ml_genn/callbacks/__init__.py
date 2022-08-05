from .callback import Callback
from .progress_bar import BatchProgressBar
from .spike_recorder import SpikeRecorder
from .var_recorder import VarRecorder

from ..utils.module import get_module_classes

default_callbacks = get_module_classes(globals(), Callback)
