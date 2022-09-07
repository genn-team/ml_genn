from .callback import Callback


class OptimiserParamSchedule(Callback):
    def __init__(self, param_name, func):
        self.param_name = param_name
        self.func = func

    def set_params(self, compiled_network, **kwargs):
        # Check optimiser has named parameter
        # **YUCK** parameter access needs to be tidied
        self._optimiser = compiled_network.optimiser

    def on_epoch_begin(self, epoch):
        # Set parameter to return value of function
        setattr(self._optimiser, self.param_name,
                self.func(epoch, getattr(self._optimiser, self.param_name)))

