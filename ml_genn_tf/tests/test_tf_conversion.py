import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

from ml_genn_tf.converters import Simple
from ml_genn.connectivity import AvgPoolConv2D, AvgPoolDense2D, Conv2D, Dense


def test_sequential_tf_conversion():
    '''
    Test Sequential TensorFlow model conversion.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # TensorFlow model
    tf_model = models.Sequential(name='test_sequential_tf_conversion')

    tf_model.add(layers.Input(shape=(32, 32, 3), name='inputs'))

    tf_model.add(layers.Conv2D(32, 3, padding='same', activation='relu', use_bias=False, name='block1_conv1'))
    tf_model.add(layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, name='block1_conv2'))
    tf_model.add(layers.AveragePooling2D(2, name='block1_pool'))

    tf_model.add(layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, name='block2_conv1'))
    tf_model.add(layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, name='block2_conv2'))
    tf_model.add(layers.AveragePooling2D(2, name='block2_pool'))

    tf_model.add(layers.Flatten())
    tf_model.add(layers.Dense(256, activation='relu', use_bias=False, name='dense1'))
    tf_model.add(layers.Dense(10, activation='relu', use_bias=False, name='dense2'))

    # ML GeNN model
    converter = Simple(500)
    mlg_net, mlg_net_inputs, mlg_net_outputs = converter.convert(tf_model)
    
    # Check that resultant model only has one input and output
    assert len(mlg_net_inputs) == 1
    assert len(mlg_net_outputs) == 1
    
    # Input layer
    mlg_pop = mlg_net_inputs[0]
    assert mlg_pop.shape == (32, 32, 3)
    assert len(mlg_pop.outgoing_connections) == 1


    # block1_conv1 population
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Conv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (32, 32, 32)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # block1_conv1 connection
    tf_layer = tf_model.get_layer('block1_conv1')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()


    # block1_conv2 population
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Conv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (32, 32, 64)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # block1_conv2 connection
    tf_layer = tf_model.get_layer('block1_conv2')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()


    # block2_conv1 population
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, AvgPoolConv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (16, 16, 64)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # block2_conv1 connection
    tf_layer = tf_model.get_layer('block2_conv1')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()


    # block2_conv2 population
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Conv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (16, 16, 64)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # block2_conv2 connection
    tf_layer = tf_model.get_layer('block2_conv2')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()


    # dense1 population
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, AvgPoolDense2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (256,)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # dense1 connection
    tf_layer = tf_model.get_layer('dense1')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()
    
    
    # dense2 population
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Dense)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (10,)
    assert len(mlg_pop.outgoing_connections) == 0
    
    # dense2 connection
    tf_layer = tf_model.get_layer('dense2')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()
    assert mlg_pop == mlg_net_outputs[0]

def test_functional_tf_conversion():
    '''
    Test Functional TensorFlow model conversion.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # TensorFlow model
    inputs =  layers.Input(shape=(32, 32, 3), name='inputs')

    b1c1 =    layers.Conv2D(32, 3, padding='same', activation='relu', use_bias=False, name='block1_conv1')(inputs)
    b1c2 =    layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, name='block1_conv2')(b1c1)
    b1p =     layers.AveragePooling2D(2, name='block1_pool')(b1c2)

    b2c1 =    layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, name='block2_conv1')(b1p)
    b2c2 =    layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, name='block2_conv2')(b2c1)
    b2p =     layers.AveragePooling2D(2, name='block2_pool')(b2c2)

    b1 =      layers.AveragePooling2D(4)(b1c2)
    b2 =      b2p
    add =     layers.add([b1, b2])

    flat =    layers.Flatten()(add)
    d1 =      layers.Dense(256, activation='relu', use_bias=False, name='dense1')(flat)
    d2 =      layers.Dense(10, activation='relu', use_bias=False, name='dense2')(d1)

    outputs = d2

    tf_model = models.Model(inputs, outputs, name='test_functional_tf_conversion')

    # ML GeNN model
    converter = Simple(500)
    mlg_net, mlg_net_inputs, mlg_net_outputs = converter.convert(tf_model)
    
    # Check that resultant model only has one input and output
    assert len(mlg_net_inputs) == 1
    assert len(mlg_net_outputs) == 1

    # Input layer
    mlg_pop = mlg_net_inputs[0]
    assert mlg_pop.shape == (32, 32, 3)
    assert len(mlg_pop.outgoing_connections) == 1


    # block1_conv1 layer
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Conv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (32, 32, 32)
    assert len(mlg_pop.outgoing_connections) == 1

    # block1_conv1 connection
    tf_layer = tf_model.get_layer('block1_conv1')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()


    # block1_conv2 layer
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Conv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (32, 32, 64)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # block1_conv2 connection
    tf_layer = tf_model.get_layer('block1_conv2')
    weights = mlg_conn.connectivity.weight
    assert weights.is_array
    assert np.equal(weights.value, tf_layer.get_weights()).all()


    # block2_conv1 layer
    mlg_conn = mlg_pop.outgoing_connections[0]()
    assert isinstance(mlg_conn.connectivity, Conv2D)
    mlg_pop = mlg_conn.target()
    assert mlg_pop.shape == (16, 16, 64)
    assert len(mlg_pop.outgoing_connections) == 1
    
    # block2_conv1 connection
    tf_layer = tf_model.get_layer('block2_conv1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block2_conv1 synapses
    synapses = mlg_layer.upstream_synapses
    assert(isinstance(synapses[0], AvgPoolConv2D))

    # block2_conv2 layer
    mlg_layer = mlg_model.layers[4]
    assert(mlg_layer.shape == (16, 16, 64))
    assert(isinstance(mlg_layer, Layer))
    # block2_conv2 weights
    tf_layer = tf_model.get_layer('block2_conv2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block2_conv2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(isinstance(synapses[0], Conv2DSynapses))

    # dense1 layer
    mlg_layer = mlg_model.layers[5]
    assert(mlg_layer.shape == (256,))
    assert(isinstance(mlg_layer, Layer))
    # dense1 weights
    tf_layer = tf_model.get_layer('dense1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # dense1 synapses
    synapses = mlg_layer.upstream_synapses.copy()
    #synapses.sort(key=lambda x: x.source().name)
    assert(synapses[0].source().name == 'block1_conv2')
    assert(isinstance(synapses[0], AvePool2DDenseSynapses))
    assert(synapses[1].source().name == 'block2_conv2')
    assert(isinstance(synapses[1], AvePool2DDenseSynapses))

    # dense2 layer
    mlg_layer = mlg_model.layers[6]
    assert(mlg_layer.shape == (10,))
    assert(isinstance(mlg_layer, Layer))
    
    # dense2 weights
    tf_layer = tf_model.get_layer('dense2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # dense2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(isinstance(synapses[0], DenseSynapses))

if __name__ == '__main__':
    test_sequential_tf_conversion()
    test_functional_tf_conversion()
