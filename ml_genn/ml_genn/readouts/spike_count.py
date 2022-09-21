import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class SpikeCount(Readout):
    def add_readout_logic(self, model: NeuronModel):
        # If model isn't spiking, give error
        if "threshold_condition_code" not in model.model:
            raise RuntimeError("SpikeCount readout can only "
                               "be used with spiking models")

        # Make copy of model
        model_copy = deepcopy(model)

        # Add code to increment spike count
        model_copy.append_reset_code("$(Scount)++;")

        # Add integer spike count variable and initialise to zero
        model_copy.add_var("Scount", "unsigned int", 0)

        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull spike count from genn
        genn_pop.pull_var_from_device("Scount")

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars["Scount"].view,
                          (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [("Scount", "unsigned int", 0)]