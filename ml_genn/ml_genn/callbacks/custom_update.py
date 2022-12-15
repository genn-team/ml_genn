from .callback import Callback


class CustomUpdate(Callback):
    def __init__(self, name):
        self.name = name
    
    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def _custom_update(self):
        self._compiled_network.genn_model.custom_update(self.name)


class CustomUpdateOnBatchBegin(CustomUpdate):
    def on_batch_begin(self, batch):
        self._custom_update()


class CustomUpdateOnBatchEnd(CustomUpdate):
    def on_batch_end(self, batch, metrics):
        self._custom_update()


class CustomUpdateOnTimestepBegin(CustomUpdate):
    def on_timestep_begin(self, batch):
        self._custom_update()


class CustomUpdateOnTimestepEnd(CustomUpdate):
    def on_timestep_end(self, batch):
        self._custom_update()