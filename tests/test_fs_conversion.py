import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

import ml_genn as mlg

def test_delay_balancing():
    '''
    Test delay-balancing when converting ResNet-style TensorFlow model to few-spike.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Define overly-complex branching network which ends up with unity gain
    inputs =  layers.Input(shape=1, name='inputs')
    
    dense_b1_1 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b1_1',
                              kernel_initializer=lambda shape, dtype: [[1.0]])(inputs)
    dense_b1_2 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b1_2',
                              kernel_initializer=lambda shape, dtype: [[1.0]])(dense_b1_1)
    dense_b1_3 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b1_3',
                              kernel_initializer=lambda shape, dtype: [[1.0]])(dense_b1_2)
    
    dense_b2_1 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b2_1',
                              kernel_initializer=lambda shape, dtype: [[1.0]])(inputs)
    dense_b2_2 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b2_2',
                              kernel_initializer=lambda shape, dtype: [[1.0]])(dense_b2_1)
    
    dense_b3_1_1 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b3_1_1',
                                kernel_initializer=lambda shape, dtype: [[1.0]])(inputs)
    dense_b3_2_1 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b3_2_1',
                                kernel_initializer=lambda shape, dtype: [[1.0]])(inputs)
    add_b3 = layers.add([dense_b3_1_1, dense_b3_2_1])
    dense_b3_2 = layers.Dense(1, activation='relu', use_bias=False, name='dense_b3_2',
                              kernel_initializer=lambda shape, dtype: [[0.5]])(add_b3)
    
    add = layers.add([dense_b1_3, dense_b2_2, dense_b3_2])
    
    output = layers.Dense(1, activation='relu', use_bias=False, name='output',
                          kernel_initializer=lambda shape, dtype: [[1.0 / 3.0]])(add)
    
    tf_model = models.Model(inputs, output, name='test_delay_balancing')
    
    # Define array of inputs and get TF model for them
    x = np.arange(0.0, 5.0).reshape((-1, 1))
    tf_y = tf_model(x).numpy() 
    
    # Check model does indeed have unity gain
    assert np.array_equal(x, tf_y)
    
    # Convert model using few spike technique
    converter = mlg.converters.FewSpike(K=8, alpha=8.0, norm_data=x)
    mlg_model = mlg.Model.convert_tf_model(tf_model, converter=converter)
    
    # Loop through inputs, taking into account pipeline depth
    pipeline_depth = mlg_model.calc_pipeline_depth()
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