import numpy as np

from .readout import TimeWindowReadout
from ..utils.model import NeuronModel

from copy import deepcopy


class FirstSpikeTime(TimeWindowReadout):
    """Read out time of first spike emitted by population.
    Spike times are negated so standard metrics and loss functions can be employed."""
    def add_readout_logic(self, model: NeuronModel, **kwargs):
        # If model isn't spiking, give error
        if "threshold_condition_code" not in model.model:
            raise RuntimeError("FirstSpikeTime readout can only "
                               "be used with spiking models")

        # Add code to record time of first spike
        # **NOTE** time is recorded with respect to t == 0 (trial start)
        model.append_reset_code(
            self.windowed_readout_code(f"TFirstSpike = fmax(-t, TFirstSpike);", **kwargs))

        # Add time of first spike variable and initialise to float min
        # **YUCK** REALLY should be timepoint but then you can't softmax
        # **YUCK** Correct minimum for scalar
        model.add_var("TFirstSpike", "scalar",
                           np.finfo(np.float32).min)

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull time of first spike from genn
        genn_pop.vars["TFirstSpike"].pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars["TFirstSpike"].view,
                          (batch_size,) + shape)

    @property
    def reset_vars(self):
        # **YUCK** REALLY should be timepoint but then you can't softmax
        # **YUCK** Correct minimum for scalar
        return [("TFirstSpike", "scalar", np.finfo(np.float32).min)]
