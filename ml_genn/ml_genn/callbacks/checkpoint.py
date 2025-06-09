from typing import Union
from .callback import Callback
from ..serialisers import Serialiser


class Checkpoint(Callback):
    """Callback which serialises network state 
    after a specified number of epochs.
    
    Args:
        serialiser:     Serialiser to use
        epoch_interval: After how many epochs should checkpoints be saved?
    """
    def __init__(self, serialiser: Union[Serialiser, str] ="numpy",
                 epoch_interval: int = 1, only_best: bool = False):
        self.serialiser = serialiser
        self.epoch_interval = epoch_interval
        self.only_best = only_best

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def on_epoch_end(self, epoch, metrics):
        # If we should checkpoint this epoch
        if (epoch % self.epoch_interval) == 0:
            if self.only_best:
                self._compiled_network.save((0,), self.serialiser)
            else:
                self._compiled_network.save((epoch,), self.serialiser)
