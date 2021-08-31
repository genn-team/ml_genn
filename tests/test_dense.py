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


def model_input_all_on():
    return np.array([
        [1, 1, 1, 1, 1],
    ], dtype=np.float64)


def model_input_some_on():
    return np.array([
        [1, 0, 1, 0, 1],
    ], dtype=np.float64)


def model_input_all_off():
    return np.array([
        [0, 0, 0, 0, 0],
    ], dtype=np.float64)


def model_weights_0():
    return np.array([
        [0, 4, -20, 1, 0, 0, 1],
        [1, 3, -10, 0, 1, 0, 1],
        [2, 2,  10, 1, 0, 0, 1],
        [3, 1,  -5, 0, 1, 0, 1],
        [4, 0,  -5, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_dense_all_on():
    '''
    Test Dense with all inputs on.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 5), dtype=np.float64)
    x[0, :] = model_input_all_on()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='output', use_bias=False, input_shape=(5,)),
    ], name='test_dense_all_on')
    tf_model.set_weights([model_weights_0()])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_dense_some_on():
    '''
    Test Dense with some inputs on.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 5), dtype=np.float64)
    x[0, :] = model_input_some_on()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='output', use_bias=False, input_shape=(5,)),
    ], name='test_dense_some_on')
    tf_model.set_weights([model_weights_0()])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_dense_all_off():
    '''
    Test Dense with all inputs off.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 5), dtype=np.float64)
    x[0, :] = model_input_all_off()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='output', use_bias=False, input_shape=(5,)),
    ], name='test_dense_all_off')
    tf_model.set_weights([model_weights_0()])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


if __name__ == '__main__':
    test_dense_all_on()
    test_dense_some_on()
    test_dense_all_off()
