import numpy as np
import tensorflow as tf
import ml_genn as mlg
import pytest


def model_compare_tf_and_mlg(tf_model, x, connectivity_type='procedural'):
    # Run TensorFlow model
    tf_y = tf_model(x).numpy()

    # Run ML GeNN model
    mlg_model = mlg.Model.convert_tf_model(tf_model, converter=mlg.converters.Simple('spike'), 
                                           connectivity_type=connectivity_type)
    mlg_model.compile(dt=1.0, batch_size=1)
    mlg_model.outputs[0].neurons.set_threshold(np.float64(np.inf))
    mlg_model.set_input_batch([x])
    mlg_model.step_time(2)

    nrn = mlg_model.outputs[0].neurons.nrn
    nrn.pull_var_from_device('Vmem')
    mlg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert np.allclose(mlg_y, tf_y, rtol=0.0, atol=1.0e-5)

    return mlg_model


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


def test_avepool2d_conv2d_in_chan_1_out_chan_1_padding_valid():
    '''
    Test AvePool2DConv2D with 1 input channel, 1 output channel and valid pool padding.
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
        tf.keras.layers.AveragePooling2D(2, padding='valid', input_shape=(12, 12, 1)),
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_1_out_chan_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


def test_avepool2d_conv2d_in_chan_1_out_chan_1_stride_3_padding_valid():
    '''
    Test AvePool2DConv2D with 1 input channel, 1 output channel and valid pool padding.
    Pool size 2, pool strides 3, conv size 3, conv strides 1.
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
        tf.keras.layers.AveragePooling2D(2, strides=3, padding='valid', input_shape=(12, 12, 1)),
        tf.keras.layers.Conv2D(1, 3, strides=1, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_1_out_chan_1_stride_3_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


def test_avepool2d_conv2d_in_chan_2_out_chan_1_padding_valid():
    '''
    Test AvePool2DConv2D with 2 input channels, 1 output channel and valid pool padding.
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
        tf.keras.layers.AveragePooling2D(2, padding='valid', input_shape=(12, 12, 2)),
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_2_out_chan_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


def test_avepool2d_conv2d_in_chan_1_out_chan_2_padding_valid():
    '''
    Test AvePool2DConv2D with 1 input channel, 2 output channels and valid pool padding.
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
        tf.keras.layers.AveragePooling2D(2, padding='valid', input_shape=(12, 12, 1)),
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_1_out_chan_2_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


def test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_valid():
    '''
    Test AvePool2DConv2D with 2 input channels, 2 output channels and valid pool padding.
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
        tf.keras.layers.AveragePooling2D(2, padding='valid', input_shape=(12, 12, 2)),
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


def test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_valid_sparse():
    '''
    Test AvePool2DConv2D with 2 input channels, 2 output channels and valid pool padding (SPARSE connectivity).
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
        tf.keras.layers.AveragePooling2D(2, padding='valid', input_shape=(12, 12, 2)),
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_valid_sparse')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x, connectivity_type='sparse')


def test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_same():
    '''
    Test AvePool2DConv2D with 2 input channels, 2 output channels and same pool padding.
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
        tf.keras.layers.AveragePooling2D(2, padding='same', input_shape=(12, 12, 2)),
        tf.keras.layers.Conv2D(2, 3, name='output', padding='same', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_same')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


def test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_same_sparse():
    '''
    Test AvePool2DConv2D with 2 input channels, 2 output channels and same pool padding (SPARSE connectivity).
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
        tf.keras.layers.AveragePooling2D(2, padding='same', input_shape=(12, 12, 2)),
        tf.keras.layers.Conv2D(2, 3, name='output', padding='same', use_bias=False),
    ], name='test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_same_sparse')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x, connectivity_type='sparse')


@pytest.mark.xfail
def test_avepool2d_conv2d_border_pool_crop():
    '''
    Test AvePool2DConv2D pool cropping at borders.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.ones((1, 12, 12, 1), dtype=np.float32)

    # Kernels
    k = np.ones((3, 3, 1, 1), dtype=np.float32)

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='same', input_shape=(12, 12, 1)),
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid', use_bias=False),
    ], name='test_avepool2d_conv2d_border_pool_crop')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, x)


if __name__ == '__main__':
    test_avepool2d_conv2d_in_chan_1_out_chan_1_padding_valid()
    test_avepool2d_conv2d_in_chan_1_out_chan_1_stride_3_padding_valid()
    test_avepool2d_conv2d_in_chan_2_out_chan_1_padding_valid()
    test_avepool2d_conv2d_in_chan_1_out_chan_2_padding_valid()
    test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_valid()
    test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_valid_sparse()
    test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_same()
    test_avepool2d_conv2d_in_chan_2_out_chan_2_padding_same_sparse()
    test_avepool2d_conv2d_border_pool_crop()
