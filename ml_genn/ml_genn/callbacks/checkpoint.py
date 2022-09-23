from .callback import Callback


class Checkpoint(Callback):
    def __init__(self, serialiser="numpy", epoch_interval=1,
                 weights=True, delays=False):
        self.serialiser = serialiser
        self.epoch_interval = epoch_interval
        self.weights = weights
        self.delays = delays

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def on_epoch_end(self, epoch, metrics):
        # If we should checkpoint this epoch
        if (epoch % epoch_interval) == 0:
            self._compiled_network.save((epoch,), self.serialiser)

