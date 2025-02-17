import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class SpikeCount(Readout):
    """Read out number of spikes emitted by population"""
    def add_readout_logic(self, model: NeuronModel, **kwargs):
        # If model isn't spiking, give error
        if "threshold_condition_code" not in model.model:
            raise RuntimeError("SpikeCount readout can only "
                               "be used with spiking models")

        # Add code to increment spike count
        model.append_reset_code("Scount++;")

        # Add integer spike count variable and initialise to zero
        model.add_var("Scount", "unsigned int", 0)

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull spike count from genn
        genn_pop.vars["Scount"].pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars["Scount"].view,
                          (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [("Scount", "unsigned int", 0)]
