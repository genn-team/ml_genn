import numpy as np
import tensorflow as tf
import tensor_genn as tg


def test_averagepooling2d_in_chan_1_stride_2_2_padding_valid_combine_dense():
    '''
    Test AveragePooling2D weight matrix with 1 input channel,
    a stride of (2, 2) and valid padding.
    '''

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(2, name='averagepooling2d', padding='valid', strides=(2, 2), input_shape=(5, 5, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, name='dense', use_bias=False),
    ], name='test_averagepooling2d_in_chan_1_stride_2_2_padding_valid_combine_dense')

    # Add downstream layer weights
    #tf_dense_w_vals = np.identity(4)
    tf_dense_w_vals = np.random.randn(4, 4)
    tf_model.set_weights([tf_dense_w_vals])

    # Convert model
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model)
    tg_model.compile()

    # Check weight matrix
    assert len(tg_model.weight_vals) == 1
    assert tg_model.weight_vals[0].shape == (25, 4)
    w_vals = tg_model.weight_vals[0]

    assert len(tg_model.weight_conn) == 1
    assert tg_model.weight_conn[0].shape == (25, 4)
    w_conn = tg_model.weight_conn[0]

    dense_w_vals = tf_dense_w_vals
    dense_w_conn = np.ones((4, 4), dtype=np.bool)

    deferred_w_vals = np.array([
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

    target_w_vals = np.dot(deferred_w_vals, dense_w_vals)
    # print('w_vals')
    # print(w_vals)
    # print('target_w_vals')
    # print(target_w_vals)
    assert np.allclose(w_vals, target_w_vals, rtol=0.0, atol=1.0e-5)

    deferred_w_conn = np.array(deferred_w_vals != 0)
    target_w_conn = np.dot(deferred_w_conn, dense_w_conn)
    # print('w_conn')
    # print(w_conn)
    # print('target_w_conn')
    # print(target_w_conn)
    assert (w_conn == target_w_conn).all()


def test_averagepooling2d_in_chan_2_stride_2_2_padding_valid_combine_dense():
    '''
    Test AveragePooling2D weight matrix with 2 input channels,
    a stride of (2, 2) and valid padding.
    '''

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(2, name='averagepooling2d', padding='valid', strides=(2, 2), input_shape=(5, 5, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, name='dense', use_bias=False),
    ], name='test_averagepooling2d_in_chan_2_stride_2_2_padding_valid_combine_dense')

    # Add downstream layer weights
    #tf_dense_w_vals = np.identity(8)
    tf_dense_w_vals = np.random.randn(8, 8)
    tf_model.set_weights([tf_dense_w_vals])

    # Convert model
    tg_model = tg.TGModel()
    tg_model.convert_tf_model(tf_model)
    tg_model.compile()

    # Check weight matrix
    assert len(tg_model.weight_vals) == 1
    assert tg_model.weight_vals[0].shape == (50, 8)
    w_vals = tg_model.weight_vals[0]

    assert len(tg_model.weight_conn) == 1
    assert tg_model.weight_conn[0].shape == (50, 8)
    w_conn = tg_model.weight_conn[0]

    dense_w_vals = tf_dense_w_vals
    dense_w_conn = np.ones((8, 8), dtype=np.bool)

    deferred_channel_w_vals = np.array([
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

    deferred_w_vals = np.zeros((50, 8), dtype=w_vals.dtype)
    deferred_w_vals[0::2, 0::2] = deferred_channel_w_vals # one-to-one in/out channel 0
    deferred_w_vals[1::2, 1::2] = deferred_channel_w_vals # one-to-one in/out channel 1
    target_w_vals = np.dot(deferred_w_vals, dense_w_vals)
    # print('w_vals')
    # print(w_vals)
    # print('target_w_vals')
    # print(target_w_vals)
    assert np.allclose(w_vals, target_w_vals, rtol=0.0, atol=1.0e-5)

    deferred_w_conn = np.array(deferred_w_vals != 0)
    target_w_conn = np.dot(deferred_w_conn, dense_w_conn)
    # print('w_conn')
    # print(w_conn)
    # print('target_w_conn')
    # print(target_w_conn)
    assert (w_conn == target_w_conn).all()


# def test_averagepooling2d_in_chan_1_stride_2_2_padding_valid_combine_conv2d():
#     '''
#     Test AveragePooling2D weight matrix with 1 input channel,
#     a stride of (2, 2) and valid padding.
#     '''

#     # Create TensorFlow model
#     tf_model = tf.keras.models.Sequential([
#         tf.keras.layers.AveragePooling2D(2, name='averagepooling2d', padding='valid', strides=(2, 2), input_shape=(10, 10, 1)),
#         tf.keras.layers.Conv2D(1, 3, name='conv2d', padding='same', use_bias=False),
#     ], name='test_averagepooling2d_in_chan_1_stride_2_2_padding_valid_combine_conv2d')

#     # Add downstream layer weights
#     #tf_conv2d_w_vals = np.ones((3, 3, 1, 1))
#     tf_conv2d_w_vals = np.random.randn(3, 3, 1, 1)
#     tf_model.set_weights([tf_conv2d_w_vals])

#     # Convert model
#     tg_model = tg.TGModel()
#     tg_model.convert_tf_model(tf_model)
#     tg_model.compile()

#     # Check weight matrix
#     assert len(tg_model.weight_vals) == 1
#     assert tg_model.weight_vals[0].shape == (25, 4)
#     w_vals = tg_model.weight_vals[0]

#     assert len(tg_model.weight_conn) == 1
#     assert tg_model.weight_conn[0].shape == (25, 4)
#     w_conn = tg_model.weight_conn[0]





#     #dense_w_vals
#     dense_w_conn = np.ones((4, 4), dtype=np.bool)

#     deferred_w_vals = np.array([
#         # input row 0
#         [0.25, 0.00, 0.00, 0.00],
#         [0.25, 0.00, 0.00, 0.00],
#         [0.00, 0.25, 0.00, 0.00],
#         [0.00, 0.25, 0.00, 0.00],
#         [0.00, 0.00, 0.00, 0.00],
#         # input row 1
#         [0.25, 0.00, 0.00, 0.00],
#         [0.25, 0.00, 0.00, 0.00],
#         [0.00, 0.25, 0.00, 0.00],
#         [0.00, 0.25, 0.00, 0.00],
#         [0.00, 0.00, 0.00, 0.00],
#         # input row 2
#         [0.00, 0.00, 0.25, 0.00],
#         [0.00, 0.00, 0.25, 0.00],
#         [0.00, 0.00, 0.00, 0.25],
#         [0.00, 0.00, 0.00, 0.25],
#         [0.00, 0.00, 0.00, 0.00],
#         # input row 3
#         [0.00, 0.00, 0.25, 0.00],
#         [0.00, 0.00, 0.25, 0.00],
#         [0.00, 0.00, 0.00, 0.25],
#         [0.00, 0.00, 0.00, 0.25],
#         [0.00, 0.00, 0.00, 0.00],
#         # input row 4
#         [0.00, 0.00, 0.00, 0.00],
#         [0.00, 0.00, 0.00, 0.00],
#         [0.00, 0.00, 0.00, 0.00],
#         [0.00, 0.00, 0.00, 0.00],
#         [0.00, 0.00, 0.00, 0.00],
#     ], dtype=w_vals.dtype)

#     target_w_vals = np.dot(deferred_w_vals, dense_w_vals)
#     # print('w_vals')
#     # print(w_vals)
#     # print('target_w_vals')
#     # print(target_w_vals)
#     assert np.allclose(w_vals, target_w_vals, rtol=0.0, atol=1.0e-5)

#     deferred_w_conn = np.array(deferred_w_vals != 0)
#     target_w_conn = np.dot(deferred_w_conn, dense_w_conn)
#     # print('w_conn')
#     # print(w_conn)
#     # print('target_w_conn')
#     # print(target_w_conn)
#     assert (w_conn == target_w_conn).all()


if __name__ == '__main__':
    test_averagepooling2d_in_chan_1_stride_2_2_padding_valid_combine_dense()
    test_averagepooling2d_in_chan_2_stride_2_2_padding_valid_combine_dense()
    #test_averagepooling2d_in_chan_1_stride_2_2_padding_valid_combine_conv2d()
