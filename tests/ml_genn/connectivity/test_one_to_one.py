import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.compilers.compiler import SupportedMatrixType
from ml_genn.neurons import IntegrateFire, BinarySpikeInput
from ml_genn.connectivity import OneToOne
from pygenn import SynapseMatrixType

@pytest.mark.parametrize(
    "size, synapse_matrix_type", 
    [(10, SynapseMatrixType.SPARSE),
     (10, SynapseMatrixType.PROCEDURAL),
     (100, SynapseMatrixType.SPARSE),
     (100, SynapseMatrixType.PROCEDURAL)])
def test_one_to_one(size, synapse_matrix_type, request):
    # Generate input tensor
    x = np.random.randint(0, 2, size=(1, size)).astype(np.float64)  

    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(BinarySpikeInput(), size)
        output = Layer(OneToOne(weight=1.0), 
                       IntegrateFire(v_thresh=np.float64(np.finfo(np.float32).max), readout="end_var"))

    compiler = InferenceCompiler(evaluate_timesteps=2)
    compiler.supported_matrix_type = SupportedMatrixType([synapse_matrix_type])
    compiled_net = compiler.compile(network, request.keywords.node.name)

    with compiled_net:
        # Evaluate ML GeNN model
        metrics, _ = compiled_net.evaluate({input: x}, {output: x},
                                           "mean_square_error")
    assert metrics[output].result < 1e-03
