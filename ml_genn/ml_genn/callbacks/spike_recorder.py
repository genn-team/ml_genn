from __future__ import annotations

import numpy as np

from .callback import Callback
from ..utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType
from ..utils.network import PopulationType

from dataclasses import dataclass
from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_pop

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compilers import CompiledNetwork

@dataclass
class State:
    compiled_network: CompiledNetwork
    data: list
    batch_mask: int
    batch_count: int = None
    pull: bool = False

class SpikeRecorder(Callback):
    """Callback used for recording spikes during simulation. 

    Args:
        pop:            Population to record from
        example_filter: Filter used to select which examples to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        neuron_filter:  Filter used to select which neurons to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        record_counts:  Should only the (per-neuron) spike count be recorded 
                        rather than all the spikes?
    """
    def __init__(self, pop: PopulationType, key=None,
                 example_filter: ExampleFilterType = None,
                 neuron_filter: NeuronFilterType = None,
                 record_counts: bool = False):
        # Get underlying population
        self._pop = get_underlying_pop(pop)

        # Stash key and whether we're recording spikes or spike-like events
        self.key = key
        self._record_counts = record_counts

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # Create neuron filter mask
        self._neuron_mask = get_neuron_filter_mask(neuron_filter,
                                                   self._pop.shape)

    def create_state(self, compiled_network, **kwargs):
        # Check number of recording timesteps ahs been set
        if compiled_network.num_recording_timesteps is None:
            raise RuntimeError("Cannot use SpikeRecorder callback "
                               "without setting num_recording_timesteps "
                               "on compiled model")

        # Check spike recording has been enabled on population
        # **YUCK** it's kinda annoying we have to do this
        genn_pop = compiled_network.neuron_populations[self._pop]
        if not genn_pop.spike_recording_enabled:
            raise RuntimeError(
                "SpikeRecorder callback can only be used to record "
                "spikes from Populations/Layers with record_spikes=True")

        # Create state with either list to hold spike counts 
        # or tuple of lists to hold spike times and IDs
        return State(compiled_network, 
                     [] if self._record_counts else ([], []),
                     np.ones(compiled_network.genn_model.batch_size,
                             dtype=bool))
                     

    def set_first(self, state):
        state.pull = True

    def on_batch_begin(self, state, batch):
        # Get mask for examples in this batch
        state.batch_mask = self._example_filter.get_batch_mask(
            batch, state.compiled_network.genn_model.batch_size)

    def on_timestep_end(self, state, timestep):
        # If spike recording buffer is full
        cn = state.compiled_network
        timestep = cn.genn_model.timestep
        if (timestep % cn.num_recording_timesteps) == 0:
            # If this is the spike recorder responsible for pulling, do so!
            if state.pull:
                cn.genn_model.pull_recording_buffers_from_device()

            # If anything should be recorded this batch
            if np.any(self._batch_mask):
                # Get GeNN population
                genn_pop = cn.neuron_populations[self._pop]

                # Get spike recording data
                data = genn_pop.spike_recording_data
                
                # Filter out batches we want
                data = [d for b, d in enumerate(data)
                        if state.batch_mask[b]]

                # If we only care about counts, calculate per-batch 
                # spike count and apply neuron mask
                if self._record_counts:
                    num = genn_pop.num_neurons
                    state.data.extend(
                        np.bincount(d[1], minlength=num)[self._neuron_mask]
                        for d in data)
                # Otherwise, if we are recording events
                else:
                    # If we're recording from all neurons, add data directly
                    if np.all(self._neuron_mask):
                        state.data[0].extend(d[0] for d in data)
                        state.data[1].extend(d[1] for d in data)
                    # Otherwise
                    else:
                        # Loop through batches
                        for d in data:
                            # Build event mask
                            mask = self._neuron_mask[d[1]]

                            # Add masked events to spikes
                            state.data[0].append(d[0][mask])
                            state.data[1].append(d[1][mask])

    def get_data(self, state):
        return self.key, state.data
