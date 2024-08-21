import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class FirstSpikeTime(Readout):
    """Read out time of first spike emitted by population"""
    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        # If model isn't spiking, give error
        if "threshold_condition_code" not in model.model:
            raise RuntimeError("FirstSpikeTime readout can only "
                               "be used with spiking models")

        # Make copy of model
        model_copy = deepcopy(model)

        # Add code to record time of first spike
        model_copy.append_reset_code(
        """
        if(t < TFirstSpike) {
            TFirstSpike = t;
        }
        """)

        # Add integer spike count variable and initialise to uint32 max
        model_copy.add_var("TFirstSpike", "unsigned int",
                           np.iinfo(np.uint32).max)

        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull time of first spike from genn
        genn_pop.vars["TFirstSpike"].pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars["TFirstSpike"].view,
                          (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [("TFirstSpike", "unsigned int", np.iinfo(np.uint32).max)]
