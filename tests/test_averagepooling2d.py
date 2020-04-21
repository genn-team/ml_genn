import numpy as np
import tensorflow as tf
import tensor_genn as tg


def model_compare_tf_and_tg(tf_model, x):
    # Run TensorFlow model
    tf_y = tf_model(x).numpy()

    # Run TensorGeNN model
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model, dt=1.0, input_type=tg.InputType.SPIKE)
    tg_model.set_inputs(x[0, :, :, :])
    tg_model.step_time(2)
    neurons = tg_model.g_model.neuron_populations['dense_nrn']
    tg_model.g_model.pull_var_from_device(neurons.name, 'Vmem_peak')
    tg_y = neurons.vars['Vmem_peak'].view.reshape(tf_y.shape)

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


def test_averagepooling2d_chan_1_stride_3_3_padding_valid():
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
    ], name='test_averagepooling2d_chan_1_stride_3_3_padding_valid')
    tf_model.set_weights([np.identity(9)])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_averagepooling2d_chan_2_stride_3_3_padding_valid():
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
    ], name='test_averagepooling2d_chan_2_stride_3_3_padding_valid')
    tf_model.set_weights([np.identity(18)])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_averagepooling2d_chan_2_stride_3_3_padding_same():
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
    ], name='test_averagepooling2d_chan_2_stride_3_3_padding_same')
    tf_model.set_weights([np.identity(32)])

    # Compare TensorFlow and TensorGeNN models
    model_compare_tf_and_tg(tf_model, x)


def test_averagepooling2d_merge_dense_chan_1_stride_2_2_padding_valid():
    '''
    Test AveragePooling2D weight matrix with 1 input channel,
    a stride of (2, 2) and valid padding.
    '''

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(2, name='averagepooling2d', padding='valid', strides=(2, 2), input_shape=(5, 5, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, name='dense', use_bias=False),
    ], name='test_averagepooling2d_merge_dense_chan_1_stride_2_2_padding_valid')

    # Prepare downstream layer
    dense_w_vals = np.identity(4)
    dense_w_conn = np.ones((4, 4))
    tf_model.set_weights([dense_w_vals])

    # Convert model
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model, dt=1.0, input_type=tg.InputType.SPIKE)

    # Check weight matrix
    assert len(tg_model.weight_vals) == 1
    w_vals = tg_model.weight_vals[0]
    assert w_vals.shape == (25, 4)

    assert len(tg_model.weight_conn) == 1
    w_conn = tg_model.weight_conn[0]
    assert w_conn.shape == (25, 4)

    target_w_vals = np.array([
        # input row 0
        [0.25, 0.00, 0.00, 0.00],
        [0.25, 0.00, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        # input row 1
        [0.25, 0.00, 0.00, 0.00],
        [0.25, 0.00, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        # input row 2
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00],
        # input row 3
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00],
        # input row 4
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
    ], dtype=w_vals.dtype)

    #print('w_vals')
    #print(w_vals)
    #print('target_w_vals')
    #print(target_w_vals)
    assert (w_vals == target_w_vals).all()

    target_w_conn = target_w_vals != 0
    target_w_conn = np.dot(target_w_conn, dense_w_conn)
    #print('w_conn')
    #print(w_conn)
    #print('target_w_conn')
    #print(target_w_conn)
    assert (w_conn == target_w_conn).all()


def test_averagepooling2d_merge_dense_chan_2_stride_2_2_padding_valid():
    '''
    Test AveragePooling2D weight matrix with 2 input channels,
    a stride of (2, 2) and valid padding.
    '''

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(2, name='averagepooling2d', padding='valid', strides=(2, 2), input_shape=(5, 5, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, name='dense', use_bias=False),
    ], name='test_averagepooling2d_merge_dense_chan_2_stride_2_2_padding_valid')

    # Prepare downstream layer
    dense_w_vals = np.identity(8)
    dense_w_conn = np.ones((8, 8))
    tf_model.set_weights([dense_w_vals])

    # Convert model
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model, dt=1.0, input_type=tg.InputType.SPIKE)

    # Check weight matrix
    assert len(tg_model.weight_vals) == 1
    w_vals = tg_model.weight_vals[0]
    assert w_vals.shape == (50, 8)

    assert len(tg_model.weight_conn) == 1
    w_conn = tg_model.weight_conn[0]
    assert w_conn.shape == (50, 8)

    target_channel_w_vals = np.array([
        # input row 0
        [0.25, 0.00, 0.00, 0.00],
        [0.25, 0.00, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        # input row 1
        [0.25, 0.00, 0.00, 0.00],
        [0.25, 0.00, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        # input row 2
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00],
        # input row 3
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00],
        # input row 4
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00],
    ], dtype=w_vals.dtype)

    target_w_vals = np.zeros((50, 8), dtype=w_vals.dtype)
    target_w_vals[0::2, 0::2] = target_channel_w_vals # one-to-one in/out channel 0
    target_w_vals[1::2, 1::2] = target_channel_w_vals # one-to-one in/out channel 1
    #print('w_vals')
    #print(w_vals)
    #print('target_w_vals')
    #print(target_w_vals)
    assert (w_vals == target_w_vals).all()

    target_w_conn = target_w_vals != 0
    target_w_conn = np.dot(target_w_conn, dense_w_conn)
    #print('w_conn')
    #print(w_conn)
    #print('target_w_conn')
    #print(target_w_conn)
    assert (w_conn == target_w_conn).all()


if __name__ == '__main__':
    test_averagepooling2d_chan_1_stride_3_3_padding_valid()
    test_averagepooling2d_chan_2_stride_3_3_padding_valid()
    test_averagepooling2d_chan_2_stride_3_3_padding_same()
    test_averagepooling2d_merge_dense_chan_1_stride_2_2_padding_valid()
    test_averagepooling2d_merge_dense_chan_2_stride_2_2_padding_valid()
