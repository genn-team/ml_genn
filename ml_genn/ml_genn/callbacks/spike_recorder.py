import numpy as np

from .callback import Callback
from ..utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType
from ..utils.network import PopulationType

from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_pop


class SpikeRecorder(Callback):
    def __init__(self, pop: PopulationType, key=None,
                 example_filter: ExampleFilterType = None,
                 neuron_filter: NeuronFilterType = None,
                 record_spike_events: bool = False):
        # Get underlying population
        self._pop = get_underlying_pop(pop)

        # Stash key and whether we're recording spikes or spike-like events
        self.key = key
        self._record_spike_events = record_spike_events

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # Create neuron filter mask
        self._neuron_mask = get_neuron_filter_mask(neuron_filter,
                                                   self._pop.shape)

        # Is this SpikeRecorder the one responsible for pulling spikes?
        self._pull = False

        # List of spike times and IDs
        self._spikes = ([], [])

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._batch_size = compiled_network.genn_model.batch_size
        self._compiled_network = compiled_network

        # Check number of recording timesteps ahs been set
        if self._compiled_network.num_recording_timesteps is None:
            raise RuntimeError("Cannot use SpikeRecorder callback "
                               "without setting num_recording_timesteps "
                               "on compiled model")

        # Check spike recording has been enabled on population
        # **YUCK** it's kinda annoying we have to do this
        genn_pop = self._compiled_network.neuron_populations[self._pop]
        if self._record_spike_events:
            if not genn_pop.spike_event_recording_enabled:
                raise RuntimeError(
                    "SpikeRecorder callback can only be used to record "
                    "spike-like events from Populations/Layers with "
                    "record_spike_events=True")
        elif not genn_pop.spike_recording_enabled:
            raise RuntimeError(
                "SpikeRecorder callback can only be used to record "
                "spikes from Populations/Layers with record_spikes=True")

        # Create default batch mask in case on_batch_begin not called
        self._batch_mask = np.ones(self._batch_size, dtype=bool)

    def set_first(self):
        self._pull = True

    def on_batch_begin(self, batch):
        # Get mask for examples in this batch
        self._batch_mask = self._example_filter.get_batch_mask(
            batch, self._batch_size)

    def on_timestep_end(self, timestep):
        # If spike recording buffer is full
        cn = self._compiled_network
        timestep = cn.genn_model.timestep
        if (timestep % cn.num_recording_timesteps) == 0:
            # If this is the spike recorder responsible for pulling, do so!
            if self._pull:
                cn.genn_model.pull_recording_buffers_from_device()

            # If anything should be recorded this batch
            if np.any(self._batch_mask):
                # Get GeNN population
                genn_pop = cn.neuron_populations[self._pop]

                # Get byte view of data
                # **NOTE** the following is a version of the PyGeNN
                # NeuronGroup._get_event_recording_data method,
                # modified to support filtering
                data = (genn_pop._spike_event_recording_data
                        if self._record_spike_events
                        else genn_pop._spike_recording_data)
                data = data.view(dtype=np.uint8)

                # Reshape into a tensor with time, batches and recording bytes
                event_recording_bytes = genn_pop._event_recording_words * 4
                data = np.reshape(data, (-1, self._batch_size,
                                         event_recording_bytes))

                # Calculate start time of recording
                dt = cn.genn_model.dT
                start_time_ms = (timestep - data.shape[0]) * dt
                if start_time_ms < 0.0:
                    raise Exception("spike_recording_data can only be "
                                    "accessed once buffer is full.")

                # Unpack data (results in one byte per bit)
                # **THINK** is there a way to avoid this step?
                data_unpack = np.unpackbits(data, axis=2,
                                            count=genn_pop.size,
                                            bitorder="little")

                # Slice out batches we want
                data_unpack = data_unpack[:, self._batch_mask, :]

                # Loop through these batches
                for b in range(data_unpack.shape[1]):
                    # Calculate indices where there are events
                    events = np.where(
                        data_unpack[:, b, self._neuron_mask] == 1)

                    # Convert event times to ms
                    event_times = start_time_ms + (events[0] * dt)

                    # Add to lists
                    self._spikes[0].append(event_times)
                    self._spikes[1].append(events[1])

    def get_data(self):
        return self.key, self._spikes
