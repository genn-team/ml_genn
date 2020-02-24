import numpy as np
import tensorflow as tf
import tensor_genn as tg


def model_test_helper(x, y, tf_model):
    # Assert TensorFlow model is correct
    tf_y = tf_model(x).numpy()
    assert (tf_y == y).all()

    # Assert Tensor GeNN model is correct
    tg_model = tg.TGModel(tf_model)
    tg_model.create_genn_model(dt=1.0, input_type='spike')
    tg_model.set_inputs(x[0, :, :, :])
    tg_model.step_time(2)
    neurons = tg_model.g_model.neuron_populations['conv2d_nrn']
    tg_model.g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
    tg_y = neurons.vars['Vmem_peak'].view.reshape(y.shape)
    assert (tg_y == y).all()


def model_kernels():
    # Format: [row, col, in_channel, out_channel]
    k = np.empty((3, 3, 2, 2), dtype=np.float32)
    k[:, :, 0, 0] = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
    k[:, :, 1, 0] = [
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ]
    k[:, :, 0, 1] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    k[:, :, 1, 1] = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
    return k


def model_inputs():
    # Format: [sample, row, col, channel]
    x = np.empty((1, 12, 12, 2), dtype=np.float32)
    x[0, :, :, 0] = [
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
    x[0, :, :, 1] = [
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return x


def test_conv2d_in_chan_1_out_chan_1_stride_1_1_padding_valid():
    '''
    Test Conv2D with 1 input channel, 1 output channel,
    a stride of (1, 1) and valid padding.
    '''

    # Kernels
    k = model_kernels()[:, :, 0:1, 0:1]

    # Inputs
    x = model_inputs()[0:1, :, :, 0:1]

    # Target Outputs
    y = np.empty((1, 10, 10, 1), dtype=np.float32)
    y[0, :, :, 0] = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        [0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
        [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
        [0, 1, 2, 1, 0, 3, 0, 1, 2, 1],
        [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ]

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, name='conv2d', padding='valid', strides=(1, 1),
                               activation='relu', use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_1_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Test model
    model_test_helper(x, y, tf_model)


def test_conv2d_in_chan_2_out_chan_1_stride_1_1_padding_valid():
    '''
    Test Conv2D with 2 input channels, 1 output channel,
    a stride of (1, 1) and valid padding.
    '''

    # Kernels
    k = model_kernels()[:, :, 0:2, 0:1]

    # Inputs
    x = model_inputs()[0:1, :, :, 0:2]

    # Target Outputs
    y = np.empty((1, 10, 10, 1), dtype=np.float32)
    y[0, :, :, 0] = [
        [6, 4, 1, 3, 6, 4, 1, 3, 6, 4],
        [0, 2, 4, 2, 1, 2, 3, 3, 1, 1],
        [4, 2, 1, 3, 3, 2, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 3, 2, 2, 3, 2, 2, 3, 2],
        [1, 4, 1, 2, 3, 2, 1, 4, 1, 2],
        [6, 2, 3, 5, 3, 2, 6, 2, 3, 5],
        [1, 3, 3, 3, 1, 5, 1, 3, 3, 3],
        [2, 3, 2, 2, 3, 2, 2, 3, 2, 2],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ]

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, name='conv2d', padding='valid', strides=(1, 1),
                               activation='relu', use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_1_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Test model
    model_test_helper(x, y, tf_model)


def test_conv2d_in_chan_1_out_chan_2_stride_1_1_padding_valid():
    '''
    Test Conv2D with 1 input channel, 2 output channels,
    a stride of (1, 1) and valid padding.
    '''

    # Kernels
    k = model_kernels()[:, :, 0:1, 0:2]

    # Inputs
    x = model_inputs()[0:1, :, :, 0:1]

    # Target Outputs
    y = np.empty((1, 10, 10, 2), dtype=np.float32)
    y[0, :, :, 0] = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        [0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
        [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
        [0, 1, 2, 1, 0, 3, 0, 1, 2, 1],
        [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ]
    y[0, :, :, 1] = [
        [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
        [0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
        [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 1, 1, 1, 2, 0, 2, 1, 1],
        [2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ]

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='conv2d', padding='valid', strides=(1, 1),
                               activation='relu', use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_2_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Test model
    model_test_helper(x, y, tf_model)


def test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid():
    '''
    Test Conv2D with 2 input channels, 2 output channels,
    a stride of (1, 1) and valid padding.
    '''

    # Kernels
    k = model_kernels()[:, :, 0:2, 0:2]

    # Inputs
    x = model_inputs()[0:1, :, :, 0:2]

    # Target Outputs
    y = np.empty((1, 10, 10, 2), dtype=np.float32)
    y[0, :, :, 0] = [
        [6, 4, 1, 3, 6, 4, 1, 3, 6, 4],
        [0, 2, 4, 2, 1, 2, 3, 3, 1, 1],
        [4, 2, 1, 3, 3, 2, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 3, 2, 2, 3, 2, 2, 3, 2],
        [1, 4, 1, 2, 3, 2, 1, 4, 1, 2],
        [6, 2, 3, 5, 3, 2, 6, 2, 3, 5],
        [1, 3, 3, 3, 1, 5, 1, 3, 3, 3],
        [2, 3, 2, 2, 3, 2, 2, 3, 2, 2],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ]
    y[0, :, :, 1] = [
        [6, 1, 1, 6, 3, 1, 4, 3, 3, 4],
        [1, 4, 2, 1, 3, 2, 2, 3, 1, 2],
        [2, 1, 2, 2, 2, 2, 1, 2, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [3, 1, 2, 2, 2, 1, 3, 1, 2, 2],
        [0, 4, 1, 3, 1, 4, 0, 4, 1, 3],
        [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
        [0, 5, 1, 4, 1, 5, 0, 5, 1, 4],
        [3, 1, 2, 2, 2, 1, 3, 1, 2, 2],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ]

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='conv2d', padding='valid', strides=(1, 1),
                               activation='relu', use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Test model
    model_test_helper(x, y, tf_model)


def test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same():
    '''
    Test Conv2D with 2 input channels, 2 output channels,
    a stride of (1, 1) and same padding.
    '''

    # Kernels
    k = model_kernels()[:, :, 0:2, 0:2]

    # Inputs
    x = model_inputs()[0:1, :, :, 0:2]

    # Target Outputs
    y = np.empty((1, 12, 12, 2), dtype=np.float32)
    y[0, :, :, 0] = [
        [2, 0, 2, 4, 2, 1, 2, 3, 3, 1, 1, 3],
        [2, 6, 4, 1, 3, 6, 4, 1, 3, 6, 4, 0],
        [2, 0, 2, 4, 2, 1, 2, 3, 3, 1, 1, 3],
        [1, 4, 2, 1, 3, 3, 2, 2, 2, 3, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 1],
        [2, 1, 4, 1, 2, 3, 2, 1, 4, 1, 2, 3],
        [0, 6, 2, 3, 5, 3, 2, 6, 2, 3, 5, 2],
        [4, 1, 3, 3, 3, 1, 5, 1, 3, 3, 3, 1],
        [0, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2],
        [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    y[0, :, :, 1] = [
        [3, 1, 2, 4, 1, 1, 4, 2, 1, 3, 2, 1],
        [2, 6, 1, 1, 6, 3, 1, 4, 3, 3, 4, 1],
        [1, 1, 4, 2, 1, 3, 2, 2, 3, 1, 2, 3],
        [2, 2, 1, 2, 2, 2, 2, 1, 2, 3, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 3, 1, 2, 2, 2, 1, 3, 1, 2, 2, 2],
        [3, 0, 4, 1, 3, 1, 4, 0, 4, 1, 3, 1],
        [0, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 3],
        [4, 0, 5, 1, 4, 1, 5, 0, 5, 1, 4, 1],
        [1, 3, 1, 2, 2, 2, 1, 3, 1, 2, 2, 2],
        [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='conv2d', padding='same', strides=(1, 1),
                               activation='relu', use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same')
    tf_model.set_weights([k])

    # Test model
    model_test_helper(x, y, tf_model)


if __name__ == '__main__':
    test_conv2d_in_chan_1_out_chan_1_stride_1_1_padding_valid()
    test_conv2d_in_chan_2_out_chan_1_stride_1_1_padding_valid()
    test_conv2d_in_chan_1_out_chan_2_stride_1_1_padding_valid()
    test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid()
    test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same()
