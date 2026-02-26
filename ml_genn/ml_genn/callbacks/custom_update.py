import logging

from typing import Callable
from .callback import Callback

logger = logging.getLogger(__name__)


class CustomUpdate(Callback):
    """Base class for callbacks that trigger a GeNN custom update.
    
    Args:
        name:       Name of custom update to trigger
        filter_fn:  Filtering function to determine which 
                    epochs/timesteps/batches to trigger on
    """
    def __init__(self, name: str, filter_fn: Callable[[int], bool] = None):
        self.name = name
        self.filter_fn = filter_fn

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def _custom_update(self, compiled_network, number):
        # If there is no filter or filter return true, launch custom update
        if self.filter_fn is None or self.filter_fn(number):
            compiled_network.genn_model.custom_update(self.name)
            return True
        else:
            return False

class CustomUpdateOnBatchBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every batch."""
    def on_batch_begin(self, state, batch):
        if self._custom_update(state, batch):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of batch {batch}")


class CustomUpdateOnBatchEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every batch."""
    def on_batch_end(self, state, batch, metrics):
        if self._custom_update(state, batch):
            logger.debug(f"Running custom update {self.name} "
                         f"at end of batch {batch}")

class CustomUpdateOnEpochBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every epoch."""
    def on_epoch_begin(self, state, epoch):
        if self._custom_update(state, epoch):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of epoch {epoch}")


class CustomUpdateOnEpochEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every epoch."""
    def on_epoch_end(self, state, epoch, metric_state):
        if self._custom_update(state, epoch):
            logger.debug(f"Running custom update {self.name} "
                         f"at end of epoch {epoch}")


class CustomUpdateOnTimestepBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every timestep."""
    def on_timestep_begin(self, state, timestep):
        if self._custom_update(state, timestep):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of timestep {timestep}")


class CustomUpdateOnTimestepEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every timestep."""
    def on_timestep_end(self, state, timestep):
        if self._custom_update(state, timestep):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of timestep {timestep}")
