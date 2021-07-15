import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

import ml_genn as mlg
from ml_genn.layers import InputLayer
from ml_genn.layers import Layer
from ml_genn.layers import DenseSynapses
from ml_genn.layers import Conv2DSynapses
from ml_genn.layers import AvePool2DDenseSynapses
from ml_genn.layers import AvePool2DConv2DSynapses


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
    mlg_model = mlg.Model.convert_tf_model(tf_model)
    assert(mlg_model.name == 'test_sequential_tf_conversion')

    # lnput layer
    mlg_layer = mlg_model.layers[0]
    assert(mlg_layer.name == 'inputs')
    assert(mlg_layer.shape == (32, 32, 3))
    assert(isinstance(mlg_layer, InputLayer))

    # block1_conv1 layer
    mlg_layer = mlg_model.layers[1]
    assert(mlg_layer.name == 'block1_conv1')
    assert(mlg_layer.shape == (32, 32, 32))
    assert(isinstance(mlg_layer, Layer))
    # block1_conv1 weights
    tf_layer = tf_model.get_layer('block1_conv1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block1_conv1 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'inputs')
    assert(isinstance(synapses[0], Conv2DSynapses))

    # block1_conv2 layer
    mlg_layer = mlg_model.layers[2]
    assert(mlg_layer.name == 'block1_conv2')
    assert(mlg_layer.shape == (32, 32, 64))
    assert(isinstance(mlg_layer, Layer))
    # block1_conv2 weights
    tf_layer = tf_model.get_layer('block1_conv2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block1_conv2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block1_conv1')
    assert(isinstance(synapses[0], Conv2DSynapses))

    # block2_conv1 layer
    mlg_layer = mlg_model.layers[3]
    assert(mlg_layer.name == 'block2_conv1')
    assert(mlg_layer.shape == (16, 16, 64))
    assert(isinstance(mlg_layer, Layer))
    # block2_conv1 weights
    tf_layer = tf_model.get_layer('block2_conv1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block2_conv1 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block1_conv2')
    assert(isinstance(synapses[0], AvePool2DConv2DSynapses))

    # block2_conv2 layer
    mlg_layer = mlg_model.layers[4]
    assert(mlg_layer.name == 'block2_conv2')
    assert(mlg_layer.shape == (16, 16, 64))
    assert(isinstance(mlg_layer, Layer))
    # block2_conv2 weights
    tf_layer = tf_model.get_layer('block2_conv2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block2_conv2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block2_conv1')
    assert(isinstance(synapses[0], Conv2DSynapses))

    # dense1 layer
    mlg_layer = mlg_model.layers[5]
    assert(mlg_layer.name == 'dense1')
    assert(mlg_layer.shape == (256,))
    assert(isinstance(mlg_layer, Layer))
    # dense1 weights
    tf_layer = tf_model.get_layer('dense1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # dense1 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block2_conv2')
    assert(isinstance(synapses[0], AvePool2DDenseSynapses))

    # dense2 layer
    mlg_layer = mlg_model.layers[6]
    assert(mlg_layer.name == 'dense2')
    assert(mlg_layer.shape == (10,))
    assert(isinstance(mlg_layer, Layer))
    # dense2 weights
    tf_layer = tf_model.get_layer('dense2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # dense2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'dense1')
    assert(isinstance(synapses[0], DenseSynapses))


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
    mlg_model = mlg.Model.convert_tf_model(tf_model)
    assert(mlg_model.name == 'test_functional_tf_conversion')

    # lnput layer
    mlg_layer = mlg_model.layers[0]
    assert(mlg_layer.name == 'inputs')
    assert(mlg_layer.shape == (32, 32, 3))
    assert(isinstance(mlg_layer, InputLayer))

    # block1_conv1 layer
    mlg_layer = mlg_model.layers[1]
    assert(mlg_layer.name == 'block1_conv1')
    assert(mlg_layer.shape == (32, 32, 32))
    assert(isinstance(mlg_layer, Layer))
    # block1_conv1 weights
    tf_layer = tf_model.get_layer('block1_conv1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block1_conv1 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'inputs')
    assert(isinstance(synapses[0], Conv2DSynapses))

    # block1_conv2 layer
    mlg_layer = mlg_model.layers[2]
    assert(mlg_layer.name == 'block1_conv2')
    assert(mlg_layer.shape == (32, 32, 64))
    assert(isinstance(mlg_layer, Layer))
    # block1_conv2 weights
    tf_layer = tf_model.get_layer('block1_conv2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block1_conv2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block1_conv1')
    assert(isinstance(synapses[0], Conv2DSynapses))

    # block2_conv1 layer
    mlg_layer = mlg_model.layers[3]
    assert(mlg_layer.name == 'block2_conv1')
    assert(mlg_layer.shape == (16, 16, 64))
    assert(isinstance(mlg_layer, Layer))
    # block2_conv1 weights
    tf_layer = tf_model.get_layer('block2_conv1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block2_conv1 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block1_conv2')
    assert(isinstance(synapses[0], AvePool2DConv2DSynapses))

    # block2_conv2 layer
    mlg_layer = mlg_model.layers[4]
    assert(mlg_layer.name == 'block2_conv2')
    assert(mlg_layer.shape == (16, 16, 64))
    assert(isinstance(mlg_layer, Layer))
    # block2_conv2 weights
    tf_layer = tf_model.get_layer('block2_conv2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # block2_conv2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'block2_conv1')
    assert(isinstance(synapses[0], Conv2DSynapses))

    # dense1 layer
    mlg_layer = mlg_model.layers[5]
    assert(mlg_layer.name == 'dense1')
    assert(mlg_layer.shape == (256,))
    assert(isinstance(mlg_layer, Layer))
    # dense1 weights
    tf_layer = tf_model.get_layer('dense1')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # dense1 synapses
    synapses = mlg_layer.upstream_synapses.copy()
    synapses.sort(key=lambda x: x.source().name)
    assert(synapses[0].source().name == 'block1_conv2')
    assert(isinstance(synapses[0], AvePool2DDenseSynapses))
    assert(synapses[1].source().name == 'block2_conv2')
    assert(isinstance(synapses[1], AvePool2DDenseSynapses))

    # dense2 layer
    mlg_layer = mlg_model.layers[6]
    assert(mlg_layer.name == 'dense2')
    assert(mlg_layer.shape == (10,))
    assert(isinstance(mlg_layer, Layer))
    # dense2 weights
    tf_layer = tf_model.get_layer('dense2')
    weights = mlg_layer.get_weights()
    assert(np.equal(weights, tf_layer.get_weights()).all())
    # dense2 synapses
    synapses = mlg_layer.upstream_synapses
    assert(synapses[0].source().name == 'dense1')
    assert(isinstance(synapses[0], DenseSynapses))


if __name__ == '__main__':
    test_sequential_tf_conversion()
    test_functional_tf_conversion()
