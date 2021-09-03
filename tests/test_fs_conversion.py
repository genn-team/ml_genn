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
    assert np.allclose(x, tf_y)
    
    converter = mlg.converters.FewSpike(K=8, alpha=1.0)
    mlg_model = mlg.Model.convert_tf_model(tf_model, converter=converter)
    
if __name__ == '__main__':
    test_delay_balancing()