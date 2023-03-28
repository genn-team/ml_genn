import logging

from .callback import Callback

logger = logging.getLogger(__name__)


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
        logger.debug(f"Running custom update {self.name} "
                     f"at start of batch {batch}")
        self._custom_update()


class CustomUpdateOnBatchEnd(CustomUpdate):
    def on_batch_end(self, batch, metrics):
        logger.debug(f"Running custom update {self.name} "
                     f"at end of batch {batch}")
        self._custom_update()


class CustomUpdateOnTimestepBegin(CustomUpdate):
    def on_timestep_begin(self, timestep):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of timestep {timestep}")
        self._custom_update()


class CustomUpdateOnTimestepEnd(CustomUpdate):
    def on_timestep_end(self, timestep):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of timestep {timestep}")
        self._custom_update()
