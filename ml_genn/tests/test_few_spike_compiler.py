import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import FewSpikeCompiler
from ml_genn.neurons import FewSpikeInput, FewSpikeRelu
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
        input = Population(FewSpikeInput(k=8, alpha=5.0), 1)
        
        dense_b1_1 = Population(hidden_neuron, 1)
        dense_b1_2 = Population(hidden_neuron, 1)
        dense_b1_3 = Population(hidden_neuron, 1)
        
        dense_b2_1 = Population(hidden_neuron, 1)
        dense_b2_2 = Population(hidden_neuron, 1)
        
        dense_b3_1 = Population(hidden_neuron, 1)
        
        output = Population(FewSpikeRelu(k=8, alpha=5.0, output="var"), 1)
        
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
    
    compiler = FewSpikeCompiler()
    compiled_net = compiler.compile(network, "test_delay_balancing")
    
    # Define array of inputs and get TF model for them
    x = np.arange(0.0, 5.0).reshape((-1, 1))
    
    with compiled_net:
        
    
    norm_data = zip(x, np.full(x.shape, np.nan))
    tf_y = tf_model(x).numpy()
    
    # Check model does indeed have unity gain
    assert np.array_equal(x, tf_y)
    
    # Convert model using few spike technique
    converter = mlg.converters.FewSpike(K=8, alpha=8.0, norm_data=[norm_data])
    mlg_model = mlg.Model.convert_tf_model(tf_model, converter=converter)
    
    # Loop through inputs, taking into account pipeline depth
    pipeline_depth = mlg_model.outputs[0].pipeline_depth
    for i in range(len(x) + pipeline_depth):
        # If there are inputs to present, set them as input batches
        if i < len(x):
            input = np.asarray([[x[i]]])
            mlg_model.set_input_batch(input)
        
        # Reset and run model for K timesteps
        mlg_model.reset()
        mlg_model.step_time(8)
        
        # If outputs should be ready, pull and compare to x
        if i >= pipeline_depth:
            nrn = mlg_model.outputs[0].neurons.nrn
            nrn.pull_var_from_device('Fx')
            assert abs(nrn.vars['Fx'].view[0] - x[i - pipeline_depth]) < 0.1
    
if __name__ == '__main__':
    test_delay_balancing()
