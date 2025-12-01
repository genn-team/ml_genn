from numbers import Number
from typing import Callable
from .callback import Callback

ScheduleCallable = Callable[[int, Number], Number]

class OptimiserParamSchedule(Callback):
    """Callback which updates an parameter on an
    :class:`..optimisers.Optimiser` every epoch based on a callable.
    
    Args:
        param_name: Name of parameter to update. Not all optimiser 
                    parameters can be changed at runtime
        func:       Callable called every epoch to determine new parameter value
    """
    def __init__(self, param_name: str, func: ScheduleCallable):
        self.param_name = param_name
        self.func = func

    def create_state(self, compiled_network, **kwargs):
        return [s for s in compiled_network.optimiser_state]

    def on_epoch_begin(self, state, epoch):
        # Set parameter to return value of function
        for s in state:
            setattr(s, self.param_name,
                    self.func(epoch, getattr(s, self.param_name)))
