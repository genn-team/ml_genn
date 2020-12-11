import numpy as np
import tensorflow as tf
import tensor_genn as tg


def model_compare_tf_and_tg(tf_model, x, connectivity_type='procedural'):
    # Run TensorFlow model
    tf_y = tf_model(x).numpy()

    # Run TensorGeNN model
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


def model_input_all_on():
    return np.array([
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)


def model_input_some_on():
    return np.array([
        [1, 0, 1, 0, 1],
    ], dtype=np.float32)


def model_input_all_off():
    return np.array([
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)


def model_weights_0():
    return np.array([
        [0, 4, -20, 1, 0, 0, 1],
        [1, 3, -10, 0, 1, 0, 1],
        [2, 2,  10, 1, 0, 0, 1],
        [3, 1,  -5, 0, 1, 0, 1],
        [4, 0,  -5, 1, 0, 0, 1],
    ], dtype=np.float32)


def test_dense_all_on():
    '''
    Test Dense with all inputs on.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 5), dtype=np.float32)
    x[0, :] = model_input_all_on()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='output', use_bias=False, input_shape=(5,)),
    ], name='test_dense_all_on')
    tf_model.set_weights([model_weights_0()])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_dense_some_on():
    '''
    Test Dense with some inputs on.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 5), dtype=np.float32)
    x[0, :] = model_input_some_on()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='output', use_bias=False, input_shape=(5,)),
    ], name='test_dense_some_on')
    tf_model.set_weights([model_weights_0()])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_dense_all_off():
    '''
    Test Dense with all inputs off.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.empty((1, 5), dtype=np.float32)
    x[0, :] = model_input_all_off()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='output', use_bias=False, input_shape=(5,)),
    ], name='test_dense_all_off')
    tf_model.set_weights([model_weights_0()])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


if __name__ == '__main__':
    test_dense_all_on()
    test_dense_some_on()
    test_dense_all_off()
