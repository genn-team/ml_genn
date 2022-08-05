import numpy as np

from itertools import chain
from typing import Sequence, Union

from ..utils.network import get_underlying_pop

class SpikeRecorder:
    def __init__(self, pop):
        # Get underlying population
        self._pop = get_underlying_pop(pop)
        
        # Should this SpikeRecorder be the one responsible for pulling spikes?
        self._pull = False

        # List of spike times and IDs
        self.spikes = ([], [])

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network
        
        # Check number of recording timesteps ahs been set
        if self._compiled_network.num_recording_timesteps is None:
            raise RuntimeError("Cannot use SpikeRecorder callback "
                               "without setting num_recording_timesteps "
                               "on compiled model")
        
        # Check spike recording has been enabled on population
        # **YUCK** it's kinda annoying we have to do this
        if not self._compiled_network.neuron_populations[self._pop].spike_recording_enabled:
            raise RuntimeError("SpikeRecorder callback can only be used"
                               "on Populations/Layers with record_spikes=True")
    def set_first(self):
        self._pull = True
    
    def on_timestep_end(self, timestep):
        # If spike recording buffer is full
        cn = self._compiled_network
        timestep = cn.genn_model.timestep
        if (timestep % cn.num_recording_timesteps) == 0:
            # If this is the spike recorder responsible for pulling, do so!
            if self._pull:
                cn.genn_model.pull_recording_buffers_from_device()

            # Get spike times and IDs
            data = cn.neuron_populations[self._pop].spike_recording_data

            # If model is batched
            if cn.genn_model.batch_size > 1:
                # Unzip batches to get seperate lists of times and IDs 
                # and extend time and ID lists with these
                times, ids = list(zip(*data))
                self.spikes[0].extend(times)
                self.spikes[1].extend(ids)
            # Otherwise, simply add times and IDs to lists
            else:
                times, ids = data
                self.spikes[0].append(times)
                self.spikes[1].append(ids)
            
            
            
        
