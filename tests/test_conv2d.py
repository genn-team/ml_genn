import numpy as np
import tensorflow as tf
import tensor_genn as tg


def model_compare_tf_and_tg(tf_model, x, synapse_type='procedural'):
    # Run TensorFlow model
    tf_y = tf_model(x).numpy()

    # Run TensorGeNN model
    tg_model = tg.Model.convert_tf_model(tf_model, input_type=tg.InputType.SPIKE, synapse_type=synapse_type)
    tg_model.compile(dt=1.0, batch_size=1)
    tg_model.outputs[0].set_threshold(np.float64(np.inf))
    tg_model.set_input_batch([x])
    tg_model.step_time(2)

    nrn = tg_model.outputs[0].nrn[0]
    nrn.pull_var_from_device('Vmem')
    tg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert np.allclose(tg_y, tf_y, rtol=0.0, atol=1.0e-5)

    return tg_model


def model_input_0():
    return np.array([
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
    ], dtype=np.float32)


def model_input_1():
    return np.array([
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
    ], dtype=np.float32)


def model_kernel_0_0():
    return np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=np.float32)


def model_kernel_1_0():
    return np.array([
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=np.float32)


def model_kernel_0_1():
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)


def model_kernel_1_1():
    return np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)


def test_conv2d_in_chan_1_out_chan_1_stride_1_1_padding_valid():
    '''
    Test Conv2D with 1 input channel, 1 output channel,
    a conv stride of (1, 1) and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 1), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()

    # Kernels
    k = np.empty((3, 3, 1, 1), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_1_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_conv2d_in_chan_2_out_chan_1_stride_1_1_padding_valid():
    '''
    Test Conv2D with 2 input channels, 1 output channel,
    a conv stride of (1, 1) and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Kernels
    k = np.empty((3, 3, 2, 1), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()
    k[:, :, 1, 0] = model_kernel_1_0()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_1_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_conv2d_in_chan_1_out_chan_2_stride_1_1_padding_valid():
    '''
    Test Conv2D with 1 input channel, 2 output channels,
    a conv stride of (1, 1) and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 1), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()

    # Kernels
    k = np.empty((3, 3, 1, 2), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()
    k[:, :, 0, 1] = model_kernel_0_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_2_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid():
    '''
    Test Conv2D with 2 input channels, 2 output channels,
    a conv stride of (1, 1) and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Kernels
    k = np.empty((3, 3, 2, 2), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()
    k[:, :, 1, 0] = model_kernel_1_0()
    k[:, :, 0, 1] = model_kernel_0_1()
    k[:, :, 1, 1] = model_kernel_1_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid_sparse():
    '''
    Test Conv2D with 2 input channels, 2 output channels,
    a conv stride of (1, 1) and valid conv padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Kernels
    k = np.empty((3, 3, 2, 2), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()
    k[:, :, 1, 0] = model_kernel_1_0()
    k[:, :, 0, 1] = model_kernel_0_1()
    k[:, :, 1, 1] = model_kernel_1_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid_sparse')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x, synapse_type='sparse')


def test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same():
    '''
    Test Conv2D with 2 input channels, 2 output channels,
    a conv stride of (1, 1) and same conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Kernels
    k = np.empty((3, 3, 2, 2), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()
    k[:, :, 1, 0] = model_kernel_1_0()
    k[:, :, 0, 1] = model_kernel_0_1()
    k[:, :, 1, 1] = model_kernel_1_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='same', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same_sparse():
    '''
    Test Conv2D with 2 input channels, 2 output channels,
    a conv stride of (1, 1) and same conv padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 12, 12, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Kernels
    k = np.empty((3, 3, 2, 2), dtype=np.float32)
    k[:, :, 0, 0] = model_kernel_0_0()
    k[:, :, 1, 0] = model_kernel_1_0()
    k[:, :, 0, 1] = model_kernel_0_1()
    k[:, :, 1, 1] = model_kernel_1_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='same', strides=(1, 1),
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same_sparse')
    tf_model.set_weights([k])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x, synapse_type='sparse')


if __name__ == '__main__':
    test_conv2d_in_chan_1_out_chan_1_stride_1_1_padding_valid()
    test_conv2d_in_chan_2_out_chan_1_stride_1_1_padding_valid()
    test_conv2d_in_chan_1_out_chan_2_stride_1_1_padding_valid()
    test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid()
    test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_valid_sparse()
    test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same()
    test_conv2d_in_chan_2_out_chan_2_stride_1_1_padding_same_sparse()
