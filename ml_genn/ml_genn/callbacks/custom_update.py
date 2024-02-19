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

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def _custom_update(self):
        self._compiled_network.genn_model.custom_update(self.name)


class CustomUpdateOnBatchBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every batch."""
    def on_batch_begin(self, batch):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of batch {batch}")
        self._custom_update()


class CustomUpdateOnBatchEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every batch."""
    def on_batch_end(self, batch, metrics):
        logger.debug(f"Running custom update {self.name} "
                     f"at end of batch {batch}")
        self._custom_update()


class CustomUpdateOnEpochBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every epoch."""
    def on_epoch_begin(self, epoch):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of epoch {epoch}")
        self._custom_update()


class CustomUpdateOnEpochEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every epoch."""
    def on_epoch_end(self, epoch, metrics):
        logger.debug(f"Running custom update {self.name} "
                     f"at end of epoch {epoch}")
        self._custom_update()


class CustomUpdateOnTimestepBegin(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the beginning of every timestep."""
    def on_timestep_begin(self, timestep):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of timestep {timestep}")
        self._custom_update()


class CustomUpdateOnTimestepEnd(CustomUpdate):
    """Callback that triggers a GeNN custom update 
    at the end of every timestep."""
    def on_timestep_end(self, timestep):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of timestep {timestep}")
        self._custom_update()

class CustomUpdateOnTrainBegin(CustomUpdate):
    def on_train_begin(self):
        logger.debug(f"Running custom update {self.name} "
                     f"at start of training")
        self._custom_update()