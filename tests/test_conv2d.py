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


def test_conv2d_in_chan_1_out_chan_1_padding_valid():
    '''
    Test Conv2D with 1 input channel, 1 output channel and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 1)).astype(np.float64)  
   
    # Kernels
    k = np.random.random_sample((3, 3, 1, 1))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid',
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_conv2d_in_chan_1_out_chan_1_padding_valid_strides_2():
    '''
    Test Conv2D with 1 input channel, 1 output channel and valid conv padding,
    with stride of 2.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 1)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 1, 1))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, strides=2, name='output', padding='valid',
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_1_padding_valid_strides_2')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])

def test_conv2d_in_chan_1_out_chan_1_padding_same_strides_2():
    '''
    Test Conv2D with 1 input channel, 1 output channel and same conv padding,
    with stride of 2.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 1)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 1, 1))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, strides=2, name='output', padding='same',
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_1_padding_same_strides_2')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])

def test_conv2d_in_chan_1_out_chan_1_padding_same_strides_2():
    '''
    Test Conv2D with 1 input channel, 1 output channel and same conv padding,
    with stride of 2.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 1)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 1, 1))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, strides=2, name='output', padding='same',
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_1_padding_same_strides_2')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])

def test_conv2d_in_chan_2_out_chan_1_padding_valid():
    '''
    Test Conv2D with 2 input channels, 1 output channel and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 2)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 2, 1))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, name='output', padding='valid',
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_1_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_conv2d_in_chan_1_out_chan_2_padding_valid():
    '''
    Test Conv2D with 1 input channel, 2 output channels and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 1)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 1, 2))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid',
                               use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_in_chan_1_out_chan_2_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_conv2d_in_chan_2_out_chan_2_padding_valid():
    '''
    Test Conv2D with 2 input channels, 2 output channels and valid conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 2)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 2, 2))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid',
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_padding_valid')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_conv2d_in_chan_2_out_chan_2_padding_valid_sparse():
    '''
    Test Conv2D with 2 input channels, 2 output channels and valid conv padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 2)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 2, 2))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='valid',
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_padding_valid_sparse')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x], connectivity_type='sparse')


def test_conv2d_in_chan_2_out_chan_2_padding_same():
    '''
    Test Conv2D with 2 input channels, 2 output channels and same conv padding.
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 2)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 2, 2))
    
    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='same',
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_padding_same')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x])


def test_conv2d_in_chan_2_out_chan_2_padding_same_sparse():
    '''
    Test Conv2D with 2 input channels, 2 output channels and same conv padding (SPARSE connectivity).
    '''

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Inputs
    x = np.random.randint(0, 2, size=(1, 12, 12, 2)).astype(np.float64)  

    # Kernels
    k = np.random.random_sample((3, 3, 2, 2))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, 3, name='output', padding='same',
                               use_bias=False, input_shape=(12, 12, 2)),
    ], name='test_conv2d_in_chan_2_out_chan_2_padding_same_sparse')
    tf_model.set_weights([k])

    # Compare TensorFlow and ML GeNN models
    model_compare_tf_and_mlg(tf_model, [x], connectivity_type='sparse')


if __name__ == '__main__':
    test_conv2d_in_chan_1_out_chan_1_padding_valid()
    test_conv2d_in_chan_1_out_chan_1_padding_valid_strides_2()
    test_conv2d_in_chan_1_out_chan_1_padding_same_strides_2()
    test_conv2d_in_chan_2_out_chan_1_padding_valid()
    test_conv2d_in_chan_1_out_chan_2_padding_valid()
    test_conv2d_in_chan_2_out_chan_2_padding_valid()
    test_conv2d_in_chan_2_out_chan_2_padding_valid_sparse()
    test_conv2d_in_chan_2_out_chan_2_padding_same()
    test_conv2d_in_chan_2_out_chan_2_padding_same_sparse()
