import numpy as np

from collections import namedtuple
from pygenn import VarAccess
from typing import Sequence, Union
from .input import Input
from .neuron import Neuron
from ..utils.data import PreprocessedSpikes 
from ..utils.model import NeuronModel

from ..utils.data import batch_spikes, calc_start_spikes

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Population

genn_model = {
    "vars": [("StartSpike", "unsigned int"),
             ("EndSpike", "unsigned int", VarAccess.READ_ONLY_DUPLICATE)],
    "extra_global_params": [("SpikeTimes", "scalar*")],
    "threshold_condition_code":
        """
        StartSpike != EndSpike && t >= SpikeTimes[StartSpike]
        """,
    "reset_code":
        """
        StartSpike++;
        """}


class SpikeInput(Neuron, Input):
    def __init__(self, max_spikes=1000000):
        super(SpikeInput, self).__init__()

        self.max_spikes = max_spikes

    def set_input(self, genn_pop, batch_size: int, shape,
                  input: Union[PreprocessedSpikes, Sequence[PreprocessedSpikes]]):
        # Batch spikes
        batched_spikes = batch_spikes(input, batch_size)

        # Get view
        start_spikes_var = genn_pop.vars["StartSpike"]
        end_spikes_var = genn_pop.vars["EndSpike"]
        spike_times_egp = genn_pop.extra_global_params["SpikeTimes"]

        # Check that spike times will fit in view, copy them and push them
        num_spikes = len(batched_spikes.spike_times) 
        assert num_spikes <= len(spike_times_egp.view)
        spike_times_egp.view[0:num_spikes] = batched_spikes.spike_times
        spike_times_egp.push_to_device()

        # Calculate start and end spike indices
        end_spikes_view[:] = batched_spikes.end_spikes
        start_spikes_var.view[:] = calc_start_spikes(batched_spikes.end_spikes)
        start_spikes_var.push_to_device()
        end_spikes_var.push_to_device()

    def get_model(self, population: "Population", dt: float, batch_size: int):
        return NeuronModel(genn_model, None, {}, 
                           {"StartSpike": 0, "EndSpike": 0},
                           {"SpikeTimes": np.empty(self.max_spikes,
                                                   dtype=np.float32)})
