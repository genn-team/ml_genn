import numpy as np

from typing import Sequence, Union

from ..utils.network import get_underlying_pop

class SpikeRecorder:
    def __init__(self, pop):
        # Get underlying population
        self._pop = get_underlying_pop(pop)
        
        # Flag to track whether simualtion is batched or not
        self._in_batch = False

        # Should this SpikeRecorder be the one responsible for pulling spikes?
        self._pull = False

        # List of spike times and IDs
        self._times = []
        self._ids = []

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

            # If we're in a batch add data to the latest batch list
            if self._in_batch:
                times, ids = zip(*data)
                self._times[-1].append(times)
                self._ids[-1].append(ids)
            # Otherwise, add data directly to variable list
            else:
                times, ids = data
                self._times.append(times)
                self._ids.append(ids)
    
    def on_batch_begin(self, batch):
        # Set flag
        self._in_batch = True

        # Add new lists for this batches spike times and ids
        self._times.append([])
        self._ids.append([])
    
    @property
    def spikes(self):
        return self._times, self._ids
            
            
            
        
