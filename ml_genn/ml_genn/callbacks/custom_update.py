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

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def _custom_update(self, number):
        # If there is no filter or filter return true, launch custom update
        if self.filter_fn is None or self.filter_fn(number):
            self._compiled_network.genn_model.custom_update(self.name)
            return True
        else:
            return False


class CustomUpdateOnBatchBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every batch."""
    def on_batch_begin(self, batch):
        if self._custom_update(batch):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of batch {batch}")


class CustomUpdateOnBatchEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every batch."""
    def on_batch_end(self, batch, metrics):
        if self._custom_update(batch):
            logger.debug(f"Running custom update {self.name} "
                         f"at end of batch {batch}")

class CustomUpdateOnEpochBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every epoch."""
    def on_epoch_begin(self, epoch):
        if self._custom_update(epoch):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of epoch {epoch}")


class CustomUpdateOnEpochEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every epoch."""
    def on_epoch_end(self, epoch, metrics):
        if self._custom_update(epoch):
            logger.debug(f"Running custom update {self.name} "
                         f"at end of epoch {epoch}")


class CustomUpdateOnTimestepBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every timestep."""
    def on_timestep_begin(self, timestep):
        if self._custom_update(timestep):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of timestep {timestep}")
        


class CustomUpdateOnTimestepEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every timestep."""
    def on_timestep_end(self, timestep):
        if self._custom_update(timestep):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of timestep {timestep}")