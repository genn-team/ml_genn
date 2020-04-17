import numpy as np
import tensorflow as tf
import tensor_genn as tg


def model_test_helper(tf_model, x):
    # Assert TensorFlow model is correct
    tf_y = tf_model(x).numpy()

    # Assert Tensor GeNN model is correct
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model, dt=1.0, input_type=tg.InputType.SPIKE)
    tg_model.set_inputs(x[0, :])
    tg_model.step_time(2)
    neurons = tg_model.g_model.neuron_populations['dense_nrn']
    tg_model.g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
    tg_y = neurons.vars['Vmem_peak'].view.reshape(tf_y.shape)
    assert (tg_y == tf_y).all()


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

    # Inputs
    x = np.empty((1, 5), dtype=np.float32)
    x[0, :] = model_input_all_on()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='dense', use_bias=False, input_shape=(5,)),
    ], name='test_dense_all_on')
    tf_model.set_weights([model_weights_0()])

    # Test model
    model_test_helper(tf_model, x)


def test_dense_some_on():
    '''
    Test Dense with some inputs on.
    '''

    # Inputs
    x = np.empty((1, 5), dtype=np.float32)
    x[0, :] = model_input_some_on()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='dense', use_bias=False, input_shape=(5,)),
    ], name='test_dense_some_on')
    tf_model.set_weights([model_weights_0()])

    # Test model
    model_test_helper(tf_model, x)


def test_dense_all_off():
    '''
    Test Dense with all inputs off.
    '''

    # Inputs
    x = np.empty((1, 5), dtype=np.float32)
    x[0, :] = model_input_all_off()

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7, name='dense', use_bias=False, input_shape=(5,)),
    ], name='test_dense_all_off')
    tf_model.set_weights([model_weights_0()])

    # Test model
    model_test_helper(tf_model, x)


if __name__ == '__main__':
    test_dense_all_on()
    test_dense_some_on()
    test_dense_all_off()
