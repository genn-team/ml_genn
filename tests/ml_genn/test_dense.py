import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, BinarySpikeInput
from ml_genn.connectivity import Dense

@pytest.mark.parametrize(
    "in_size, out_size",
    [(5, 5),
     (5, 10)])
def test_dense(in_size, out_size, request):
    x = np.random.randint(0, 2, size=(1, in_size)).astype(np.float64)

    # Generate and set weights
    w = np.random.random_sample((in_size, out_size))
    
    # Calculate matrix product
    y_true = np.matmul(x, w)

    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(BinarySpikeInput(), in_size)
        output = Layer(Dense(weight=w), 
                       IntegrateFire(v_thresh=np.float64(np.finfo(np.float32).max), readout="var"),
                       out_size)

    compiler = InferenceCompiler(evaluate_timesteps=2)
    compiled_net = compiler.compile(network, request.keywords.node.name)

    with compiled_net:
        # Evaluate ML GeNN model
        metrics, _ = compiled_net.evaluate({input: x}, {output: y_true},
                                           "mean_square_error")
    assert metrics[output].result < 1e-03
