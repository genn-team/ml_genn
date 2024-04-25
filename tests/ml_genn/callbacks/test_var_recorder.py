import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import VarRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.neurons import BinarySpikeInput, IntegrateFire

@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("neuron_filter", [None, slice(0, None, 2)])
@pytest.mark.parametrize("example_filter", [None, range(0, 5, 2)])
def test_recording(batch_size, neuron_filter, example_filter, request):
    # Create random input for 5 examples and 10 neurons
    x = np.random.random_sample((5, 10))
    y_true = np.zeros((5, 10))
    
    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(BinarySpikeInput(), 10)
        output = Layer(Dense(weight=0), 
                       IntegrateFire(readout="var"),
                       10)

    compiler = InferenceCompiler(evaluate_timesteps=2, batch_size=batch_size)
    compiled_net = compiler.compile(network, request.keywords.node.name)

    with compiled_net:
        # Evaluate ML GeNN model
        _, cb_data = compiled_net.evaluate(
            {input: x}, {output: y_true}, "mean_square_error",
            callbacks=[VarRecorder(input, genn_var="Input", key="in",
                                   neuron_filter=neuron_filter,
                                   example_filter=example_filter)])
        
        # Loop through examples
        examples = range(5) if example_filter is None else example_filter
        for i, e in enumerate(examples):
            # Check recorded value in all timesteps matches input
            if neuron_filter is None:
                assert np.allclose(x[e], cb_data["in"][i])
            else:
                assert np.allclose(x[e][neuron_filter], cb_data["in"][i])