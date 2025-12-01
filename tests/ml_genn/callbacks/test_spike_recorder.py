import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import SpikeRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import LeakyIntegrate, SpikeInput
from ml_genn.connectivity import OneToOne

from ml_genn.utils.data import preprocess_spikes

@pytest.mark.parametrize(
    "neuron_inds", [np.arange(10), np.arange(0, 10, 2)])
def test_spike_record(neuron_inds):
    # Generate a spike for the first id timesteps for neuron id
    input_times = []
    input_ids = []
    for i in range(10):
        input_times.append(np.arange(i))
        input_ids.append(np.ones(i, dtype=int) * i)
    input_times = np.concatenate(input_times)
    input_ids = np.concatenate(input_ids)
    
    # Preprocess
    input_spikes = preprocess_spikes(input_times, input_ids, num_neurons=10)
    
    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(SpikeInput(max_spikes=len(input_times)), 10, record_spikes=True)
        output = Layer(OneToOne(weight=1.0), 
                       LeakyIntegrate(readout="var"))

    compiler = InferenceCompiler(evaluate_timesteps=10, dt=1.0)
    compiled_net = compiler.compile(network, "test_spike_record")

    with compiled_net:
        # Evaluate ML GeNN model, recording input spikes and counts 
        callbacks = [SpikeRecorder(input, key="input_spikes",
                                   neuron_filter=neuron_inds),
                     SpikeRecorder(input, key="input_spike_count",
                                   neuron_filter=neuron_inds,
                                   record_counts=True)]
        _, cb_data = compiled_net.evaluate({input: [input_spikes]}, {output: [np.zeros(10)]},
                                           "mean_square_error", callbacks=callbacks)
        
        # Check spike counts match
        assert np.array_equal(cb_data["input_spike_count"][0], np.arange(10)[neuron_inds])
        
        # Order recorded times and IDs into same order as input
        record_times = cb_data["input_spikes"][0][0]
        record_ids = cb_data["input_spikes"][1][0]
        record_order = np.lexsort((record_times, record_ids))
        
        # Check times and ids match input adter applying neuron filter
        input_masked = np.isin(input_ids, neuron_inds)
        assert np.array_equal(record_ids[record_order], input_ids[input_masked])
        assert np.array_equal(record_times[record_order], input_times[input_masked])
