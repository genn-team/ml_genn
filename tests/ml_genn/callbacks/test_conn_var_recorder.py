import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import ConnVarRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.neurons import BinarySpikeInput, IntegrateFire

@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("src_neuron_filter", [None, slice(0, None, 2)])
@pytest.mark.parametrize("trg_neuron_filter", [None, slice(0, None, 2)])
@pytest.mark.parametrize("example_filter", [None, range(0, 5, 2)])
def test_conn_var_recorder(batch_size, src_neuron_filter, trg_neuron_filter,
                           example_filter, request):
    x = np.zeros((5, 10))

    weight = np.outer(np.arange(10.0), np.arange(10.0))

    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(BinarySpikeInput(), 10)
        output = Layer(Dense(weight=weight), 
                       IntegrateFire(readout="var"))

    compiler = InferenceCompiler(evaluate_timesteps=2, batch_size=batch_size)
    compiled_net = compiler.compile(network, request.keywords.node.name)

    with compiled_net:
        # Evaluate ML GeNN model
        _, cb_data = compiled_net.evaluate(
            {input: x}, {output: x}, "mean_square_error",
            callbacks=[ConnVarRecorder(output, genn_var="g", key="weight",
                                       src_neuron_filter=src_neuron_filter,
                                       trg_neuron_filter=trg_neuron_filter,
                                       example_filter=example_filter)])

        # Check callback data contains right number of recordings
        examples = range(5) if example_filter is None else example_filter
        assert len(cb_data["weight"]) == len(examples)
        
        weight_filt = weight
        if src_neuron_filter is not None:
            weight_filt = weight_filt[src_neuron_filter]
        if trg_neuron_filter is not None:
            weight_filt = weight_filt[:,trg_neuron_filter]
        
        # Loop through examples
        for i, e in enumerate(examples):
            assert(np.allclose(cb_data["weight"][i],
                               np.broadcast_to(weight_filt, 
                                               (2,) + weight_filt.shape)))
