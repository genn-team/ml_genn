import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, BinarySpikeInput
from ml_genn.connectivity import OneToOne

@pytest.mark.parametrize(
    "size, prefer_in_memory_connect", 
    [(10, True),
     (10, False),
     (100, True),
     (100, False)])

def test_one_to_one(size, prefer_in_memory_connect, request):
    # Generate input tensor
    x = np.random.randint(0, 2, size=(1, size)).astype(np.float64)  

    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(BinarySpikeInput(), size)
        output = Layer(OneToOne(weight=1.0), 
                       IntegrateFire(v_thresh=np.float64(np.finfo(np.float32).max), output="var"))

    compiler = InferenceCompiler(evaluate_timesteps=2)
    compiled_net = compiler.compile(network, request.keywords.node.name)

    with compiled_net:
        # Evaluate ML GeNN model
        accuracy = compiled_net.evaluate_numpy({input: x},
                                               {output: x},
                                               "mean_square_error")
    assert accuracy[output].result < 1e-03
