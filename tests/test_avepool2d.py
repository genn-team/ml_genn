import numpy as np
import tensorflow as tf
import ml_genn as mlg
from converter import Converter


def model_compare_tf_and_mlg(tf_model, x, connectivity_type='procedural'):
    # Run TensorFlow model
    tf_y = tf_model(x).numpy()

    # Run ML GeNN model
    mlg_model = mlg.Model.convert_tf_model(tf_model, converter=Converter(),
                                           connectivity_type=connectivity_type,
                                           dt=1.0, batch_size=1)
    mlg_model.outputs[0].neurons.set_threshold(np.float64(np.inf))
    mlg_model.set_input_batch(x)
    mlg_model.step_time(2)

    nrn = mlg_model.outputs[0].neurons.nrn
    nrn.pull_var_from_device('Vmem')
    mlg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert(np.allclose(mlg_y, tf_y, atol=0.0, rtol=1.0e-3))

    return mlg_model


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
    ], dtype=np.float64)


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
    ], dtype=np.float64)


def test_avepool2d_in_chan_1_padding_valid():
    '''
    Test AvePool2D with 1 input channel, 1 output channel and valid pool padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 1), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='valid', input_shape=(10, 10, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.ReLU(),
    ], name='test_avepool2d_in_chan_1_padding_valid')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_avepool2d_in_chan_2_padding_valid():
    '''
    Test AvePool2D with 2 input channels, 2 output channels and valid pool padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='valid', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.ReLU(),
    ], name='test_avepool2d_in_chan_2_padding_valid')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_avepool2d_in_chan_2_padding_valid_sparse():
    '''
    Test AvePool2D with 2 input channels, 2 output channels and valid pool padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='valid', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.ReLU(),
    ], name='test_avepool2d_in_chan_2_padding_valid_sparse')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x], connectivity_type='sparse')


def test_avepool2d_in_chan_2_padding_same():
    '''
    Test AvePool2D with 2 input channels, 2 output channels and same pool padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='same', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.ReLU(),
    ], name='test_avepool2d_in_chan_2_padding_same')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_avepool2d_in_chan_2_padding_same_sparse():
    '''
    Test AvePool2D with 2 input channels, 2 output channels and same pool padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(3, padding='same', input_shape=(10, 10, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.ReLU(),
    ], name='test_avepool2d_in_chan_2_padding_same_sparse')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x], connectivity_type='sparse')


def test_avepool2d_inputs_2():
    '''
    Test AvePool2D with 2 input layers.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x0 = np.empty((1, 10, 10, 1), dtype=np.float64)
    x0[0, :, :, 0] = model_input_0()
    x1 = np.empty((1, 10, 10, 1), dtype=np.float64)
    x1[0, :, :, 0] = model_input_1()

    # Create TensorFlow model
    in0 = tf.keras.layers.Input(shape=(10, 10, 1))
    in1 = tf.keras.layers.Input(shape=(10, 10, 1))
    add = tf.keras.layers.Add()([in0, in1])
    pool = tf.keras.layers.AveragePooling2D(3, padding='valid')(add)
    flat = tf.keras.layers.Flatten()(pool)
    relu = tf.keras.layers.ReLU()(flat)
    tf_model = tf.keras.models.Model([in0, in1], [relu], name='test_avepool2d_inputs_2')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x0, x1])


def test_global_avepool2d_in_chan_1():
    '''
    Test global AvePool2D with 1 input channel and 1 output channel.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 1), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(10, 10, 1)),
        tf.keras.layers.ReLU(),
    ], name='test_global_avepool2d_in_chan_1')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_global_avepool2d_in_chan_2():
    '''
    Test global AvePool2D with 2 input channels and 2 output channels.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 10, 10, 2), dtype=np.float64)
    x[0, :, :, 0] = model_input_0()
    x[0, :, :, 1] = model_input_1()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(10, 10, 2)),
        tf.keras.layers.ReLU(),
    ], name='test_global_avepool2d_in_chan_2')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_global_avepool2d_inputs_2():
    '''
    Test global AvePool2D with 2 input layers.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x0 = np.empty((1, 10, 10, 1), dtype=np.float64)
    x0[0, :, :, 0] = model_input_0()
    x1 = np.empty((1, 10, 10, 1), dtype=np.float64)
    x1[0, :, :, 0] = model_input_1()

    # Create TensorFlow model
    in0 = tf.keras.layers.Input(shape=(10, 10, 1))
    in1 = tf.keras.layers.Input(shape=(10, 10, 1))
    add = tf.keras.layers.Add()([in0, in1])
    pool = tf.keras.layers.GlobalAveragePooling2D()(add)
    relu = tf.keras.layers.ReLU()(pool)
    tf_model = tf.keras.models.Model([in0, in1], [relu], name='test_global_avepool2d_inputs_2')

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x0, x1])


if __name__ == '__main__':
    test_avepool2d_in_chan_1_padding_valid()
    test_avepool2d_in_chan_2_padding_valid()
    test_avepool2d_in_chan_2_padding_valid_sparse()
    test_avepool2d_in_chan_2_padding_same()
    test_avepool2d_in_chan_2_padding_same_sparse()
    test_avepool2d_inputs_2()
    test_global_avepool2d_in_chan_1()
    test_global_avepool2d_in_chan_2()
    test_global_avepool2d_inputs_2()
