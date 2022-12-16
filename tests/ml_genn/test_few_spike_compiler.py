import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import FewSpikeCompiler
from ml_genn.neurons import FewSpikeReluInput, FewSpikeRelu
from ml_genn.connectivity import Dense

def test_delay_balancing():
    '''
    Test delay-balancing when converting ResNet-style TensorFlow model to few-spike.
    '''
    
    hidden_neuron = FewSpikeRelu(k=8, alpha=5.0)
    hidden_connectivity = Dense(1.0)
    output_connectivity = Dense(1.0 / 3.0)
    
    network = Network()
    with network:
        input = Population(FewSpikeReluInput(k=8, alpha=5.0), 1)
        
        dense_b1_1 = Population(hidden_neuron, 1)
        dense_b1_2 = Population(hidden_neuron, 1)
        dense_b1_3 = Population(hidden_neuron, 1)
        
        dense_b2_1 = Population(hidden_neuron, 1)
        dense_b2_2 = Population(hidden_neuron, 1)
        
        dense_b3_1 = Population(hidden_neuron, 1)
        
        output = Population(FewSpikeRelu(k=8, alpha=5.0, readout="var"), 1)
        
        # Connect input to output via block 1
        Connection(input, dense_b1_1, hidden_connectivity)
        Connection(dense_b1_1, dense_b1_2, hidden_connectivity)
        Connection(dense_b1_2, dense_b1_3, hidden_connectivity)
        Connection(dense_b1_3, output, output_connectivity)
        
        # Connect input to output via block 2
        Connection(input, dense_b2_1, hidden_connectivity)
        Connection(dense_b2_1, dense_b2_2, hidden_connectivity)
        Connection(dense_b2_2, output, output_connectivity)
        
        # Connect input to output via block 3
        Connection(input, dense_b3_1, hidden_connectivity)
        Connection(dense_b3_1, output, output_connectivity)
    
    compiler = FewSpikeCompiler(k=8)
    compiled_net = compiler.compile(network, "test_delay_balancing",
                                    inputs=input, outputs=output)
    
    # Define array of inputs and get TF model for them
    x = np.arange(0.0, 5.0).reshape((-1, 1))

    with compiled_net:
        # Evaluate model on x, calculate mean square error with x
        metrics, _ = compiled_net.evaluate({input: x}, {output: x},
                                           "mean_square_error")
        assert metrics[output].result < 0.1
    
if __name__ == '__main__':
    test_delay_balancing()
