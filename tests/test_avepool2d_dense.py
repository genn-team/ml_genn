import numpy as np
import tensorflow as tf
import ml_genn as tg


def model_compare_tf_and_tg(tf_model, x, connectivity_type='procedural'):
    # Run TensorFlow model
    tf_y = tf_model(x).numpy()

    # Run ML GeNN model
    tg_model = tg.Model.convert_tf_model(tf_model, input_type='spike', connectivity_type=connectivity_type)
    tg_model.compile(dt=1.0, batch_size=1)
    tg_model.outputs[0].neurons.set_threshold(np.float64(np.inf))
    tg_model.set_input_batch([x])
    tg_model.step_time(2)

    nrn = tg_model.outputs[0].neurons.nrn[0]
    nrn.pull_var_from_device('Vmem')
    tg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert np.allclose(tg_y, tf_y, rtol=0.0, atol=1.0e-5)

    return tg_model


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


def test_avepool2d_dense_in_chan_1_padding_valid():
    '''
    Test AvePool2DDense with 1 input channel, 1 output channel and valid pool padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 1), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='valid', input_shape=(10, 10, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(9, name='output', use_bias=False),
    ], name='test_avepool2d_dense_in_chan_1_padding_valid')
    tf_model.set_weights([np.identity(9)])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_avepool2d_dense_in_chan_2_padding_valid():
    '''
    Test AvePool2DDense with 2 input channels, 2 output channels and valid pool padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='valid', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(18, name='output', use_bias=False),
    ], name='test_avepool2d_dense_in_chan_2_padding_valid')
    tf_model.set_weights([np.identity(18)])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_avepool2d_dense_in_chan_2_padding_valid_sparse():
    '''
    Test AvePool2DDense with 2 input channels, 2 output channels and valid pool padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='valid', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(18, name='output', use_bias=False),
    ], name='test_avepool2d_dense_in_chan_2_padding_valid_sparse')
    tf_model.set_weights([np.identity(18)])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_tg(tf_model, x, connectivity_type='sparse')


def test_avepool2d_dense_in_chan_2_padding_same():
    '''
    Test AvePool2DDense with 2 input channels, 2 output channels and same pool padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='same', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, name='output', use_bias=False),
    ], name='test_avepool2d_dense_in_chan_2_padding_same')
    tf_model.set_weights([np.identity(32)])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_avepool2d_dense_in_chan_2_padding_same_sparse():
    '''
    Test AvePool2DDense with 2 input channels, 2 output channels and same pool padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float32)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='same', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, name='output', use_bias=False),
    ], name='test_avepool2d_dense_in_chan_2_padding_same_sparse')
    tf_model.set_weights([np.identity(32)])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_tg(tf_model, x, connectivity_type='sparse')


if __name__ == '__main__':
    test_avepool2d_dense_in_chan_1_padding_valid()
    test_avepool2d_dense_in_chan_2_padding_valid()
    test_avepool2d_dense_in_chan_2_padding_valid_sparse()
    test_avepool2d_dense_in_chan_2_padding_same()
    test_avepool2d_dense_in_chan_2_padding_same_sparse()
