import logging

from .callback import Callback

logger = logging.getLogger(__name__)


class CustomUpdate(Callback):
    """Base class for callbacks that trigger a GeNN custom update.
    
    Args:
        name:   Name of custom update to trigger
    """
    def __init__(self, name: str):
        self.name = name

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def _custom_update(self, compiled_network):
        compiled_network.genn_model.custom_update(self.name)


class CustomUpdateOnBatchBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every batch."""
    def on_batch_begin(self, state, batch):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of batch {batch}")
        self._custom_update(state)


class CustomUpdateOnBatchEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every batch."""
    def on_batch_end(self, state, batch, metrics):
        logger.debug(f"Running custom update {self.name} "
                     f"at end of batch {batch}")
        self._custom_update(state)


class CustomUpdateOnEpochBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every epoch."""
    def on_epoch_begin(self, state, epoch):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of epoch {epoch}")
        self._custom_update(state)


class CustomUpdateOnEpochEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every epoch."""
    def on_epoch_end(self, state, epoch, metrics):
        logger.debug(f"Running custom update {self.name} "
                     f"at end of epoch {epoch}")
        self._custom_update(state)


class CustomUpdateOnTimestepBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every timestep."""
    def on_timestep_begin(self, state, timestep):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of timestep {timestep}")
        self._custom_update(state)


class CustomUpdateOnTimestepEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every timestep."""
    def on_timestep_end(self, state, timestep):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of timestep {timestep}")
        self._custom_update(state)
