import numpy as np
import tensorflow as tf
import tensor_genn as tg


def model_test_helper(tf_model, x):
    # Assert TensorFlow model is correct
    tf_y = tf_model(x).numpy()

    # Assert Tensor GeNN model is correct
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model, dt=1.0, input_type=tg.InputType.SPIKE)
    tg_model.set_inputs(x[0, :, :, :])
    tg_model.step_time(2)
    neurons = tg_model.g_model.neuron_populations['dense_nrn']
    tg_model.g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
    tg_y = neurons.vars['Vmem_peak'].view.reshape(tf_y.shape)
    assert (tg_y == tf_y).all()


def model_input_0():
    return np.array([
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)


def model_input_1():
    return np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)


def test_averagepooling2d_in_chan_1_out_chan_1_stride_3_3_padding_valid():
    '''
    Test AveragePooling2D with 1 input channel, 1 output channel,
    a stride of (3, 3) and valid padding.
    '''

    # Inputs
    x = np.empty((1, 10, 10, 1), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, name='averagepooling2d', padding='valid', strides=(3, 3), input_shape=(10, 10, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(9, name='dense', use_bias=False),
    ], name='test_averagepooling2d_in_chan_1_out_chan_1_stride_3_3_padding_valid')
    tf_model.set_weights([np.identity(9)])

    # Test model
    model_test_helper(tf_model, x)


def test_averagepooling2d_in_chan_2_out_chan_2_stride_3_3_padding_valid():
    '''
    Test AveragePooling2D with 2 input channels, 2 output channels,
    a stride of (3, 3) and valid padding.
    '''

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, name='averagepooling2d', padding='valid', strides=(3, 3), input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(18, name='dense', use_bias=False),
    ], name='test_averagepooling2d_in_chan_2_out_chan_2_stride_3_3_padding_valid')
    tf_model.set_weights([np.identity(18)])

    # Test model
    model_test_helper(tf_model, x)


def test_averagepooling2d_in_chan_2_out_chan_2_stride_3_3_padding_same():
    '''
    Test AveragePooling2D with 2 input channels, 2 output channels,
    a stride of (3, 3) and same padding.
    '''

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, name='averagepooling2d', padding='same', strides=(3, 3), input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, name='dense', use_bias=False),
    ], name='test_averagepooling2d_in_chan_2_out_chan_2_stride_3_3_padding_same')
    tf_model.set_weights([np.identity(32)])

    # Test model
    model_test_helper(tf_model, x)


if __name__ == '__main__':
    test_averagepooling2d_in_chan_1_out_chan_1_stride_3_3_padding_valid()
    test_averagepooling2d_in_chan_2_out_chan_2_stride_3_3_padding_valid()
    test_averagepooling2d_in_chan_2_out_chan_2_stride_3_3_padding_same()
