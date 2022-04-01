import numpy as np
import tensorflow as tf
import ml_genn as mlg
import pytest
from converter import Converter

@pytest.mark.parametrize(
    'in_size, in_chan, out_chan, pool_size, pool_strides, conv_size, conv_strides, conv_padding, connect', [
        (20, 1, 1, 2, 2, 3, 1, 'valid', 'sparse'),
        (20, 1, 1, 2, 2, 3, 1, 'valid', 'procedural'),
        (20, 1, 1, 2, 2, 3, 1, 'valid', 'toeplitz'),
        (20, 2, 2, 2, 2, 3, 1, 'valid', 'sparse'),
        (20, 2, 2, 2, 2, 3, 1, 'valid', 'procedural'),
        (20, 2, 2, 2, 2, 3, 1, 'valid', 'toeplitz'),
        (20, 1, 1, 2, 2, 3, 1, 'same', 'sparse'),
        (20, 1, 1, 2, 2, 3, 1, 'same', 'procedural'),
        (20, 1, 1, 2, 2, 3, 1, 'same', 'toeplitz'),
        (20, 2, 2, 2, 2, 3, 1, 'same', 'sparse'),
        (20, 2, 2, 2, 2, 3, 1, 'same', 'procedural'),
        (20, 2, 2, 2, 2, 3, 1, 'same', 'toeplitz'),
        (20, 1, 1, 3, 3, 4, 1, 'valid', 'sparse'),
        (20, 1, 1, 3, 3, 4, 1, 'valid', 'procedural'),
        (20, 1, 1, 3, 3, 4, 1, 'valid', 'toeplitz'),
        (20, 1, 1, 3, 4, 4, 1, 'valid', 'sparse'),
        (20, 1, 1, 3, 4, 4, 1, 'valid', 'procedural'),
        (20, 1, 1, 3, 4, 4, 1, 'valid', 'toeplitz'),
        (20, 1, 1, 3, 3, 4, 1, 'same', 'sparse'),
        (20, 1, 1, 3, 3, 4, 1, 'same', 'procedural'),
        #(20, 1, 1, 3, 3, 4, 1, 'same', 'toeplitz'), # FAIL: toeplitz, even size conv kernel, same padding
        (20, 1, 1, 3, 4, 4, 1, 'same', 'sparse'),
        (20, 1, 1, 3, 4, 4, 1, 'same', 'procedural'),
        #(20, 1, 1, 3, 4, 4, 1, 'same', 'toeplitz'), # FAIL: toeplitz, even size conv kernel, same padding
    ])

def test_avepool2d_conv2d(
        in_size, in_chan, out_chan, pool_size, pool_strides,
        conv_size, conv_strides, conv_padding, connect, request):
    # Don't use all GPU memory for TF!
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Generate input tensor
    x = np.random.randint(0, 2, size=(1, in_size, in_size, in_chan)).astype(np.float64)

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(
            pool_size, strides=pool_strides, padding='valid',
            input_shape=(in_size, in_size, in_chan)),
        tf.keras.layers.Conv2D(
            out_chan, conv_size, strides=conv_strides, padding=conv_padding, use_bias=False),
    ], name=request.keywords.node.name)

    # Generate and set weights
    w = np.random.random_sample((conv_size, conv_size, in_chan, out_chan))
    tf_model.set_weights([w])

    # Run TensorFlow model
    tf_y = tf_model([x]).numpy()

    # Run ML GeNN model
    mlg_model = mlg.Model.convert_tf_model(
        tf_model, converter=Converter(), connectivity_type=connect)
    mlg_model.outputs[0].neurons.set_threshold(np.float64(np.inf))
    mlg_model.set_input_batch([x])
    mlg_model.step_time(2)

    nrn = mlg_model.outputs[0].neurons.nrn
    nrn.pull_var_from_device('Vmem')
    mlg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert(np.allclose(mlg_y, tf_y, atol=0.0, rtol=1.0e-3))