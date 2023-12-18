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
    "var_name_types": [("StartSpike", "unsigned int"),
                       ("EndSpike", "unsigned int",
                        VarAccess.READ_ONLY_DUPLICATE)],
    "extra_global_params": [("SpikeTimes", "scalar*")],
    "threshold_condition_code":
        """
        StartSpike != EndSpike && t >= SpikeTimes[StartSpike]
        """,
    "reset_code":
        """
        StartSpike++;
        """,            
    "is_auto_refractory_required": False}


class SpikeInput(Neuron, Input):
    def __init__(self, max_spikes=1000000):
        super(SpikeInput, self).__init__()

        self.max_spikes = max_spikes

    def set_input(self, genn_pop, batch_size: int, shape,
                  input: Union[PreprocessedSpikes, Sequence[PreprocessedSpikes]]):
        # Batch spikes
        batched_spikes = batch_spikes(input, batch_size)

        # Get view
        start_spikes_view = genn_pop.vars["StartSpike"].view
        end_spikes_view = genn_pop.vars["EndSpike"].view
        spike_times_view = genn_pop.extra_global_params["SpikeTimes"].view

        # Check that spike times will fit in view, copy them and push them
        num_spikes = len(batched_spikes.spike_times) 
        assert num_spikes <= len(spike_times_view)
        spike_times_view[0:num_spikes] = batched_spikes.spike_times
        genn_pop.push_extra_global_param_to_device("SpikeTimes")

        # Calculate start and end spike indices
        end_spikes_view[:] = batched_spikes.end_spikes
        start_spikes_view[:] = calc_start_spikes(batched_spikes.end_spikes)
        genn_pop.push_var_to_device("StartSpike")
        genn_pop.push_var_to_device("EndSpike")

    def get_model(self, population: "Population", dt: float, batch_size: int):
        return NeuronModel(genn_model, None, {}, 
                           {"StartSpike": 0, "EndSpike": 0},
                           {"SpikeTimes": np.empty(self.max_spikes,
                                                   dtype=np.float32)})
