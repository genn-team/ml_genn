import pytest
import numpy as np
import tensorflow as tf
import tensor_genn as tg

def test_conv2d_1_in_chan_1_out_chan_1_stride_valid():

    # Kernel
    k = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=np.float32)

    # Input
    x = np.array([
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

    # Target Output
    y = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        [0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
        [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
        [0, 1, 2, 1, 0, 3, 0, 1, 2, 1],
        [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    ], dtype=np.float32)

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, padding='valid', strides=1, activation='relu', use_bias=False, input_shape=(12, 12, 1)),
    ], name='test_conv2d_1_in_chan_1_out_chan_1_stride_valid')
    tf_k = k[:, :, np.newaxis, np.newaxis]
    tf_model.set_weights([tf_k])

    # Assert TensorFlow model is correct
    tf_x = x[np.newaxis, :, :, np.newaxis]
    tf_y = tf_model(tf_x).numpy()
    assert (tf_y[0, :, :, 0] == y).all()

    # Create Tensor GeNN model
    tg_model = tg.TGModel(tf_model)
    tg_model.create_genn_model(dt=1.0, input_type='if')

    # Assert Tensor GeNN model is correct
    neurons = tg_model.g_model.neuron_populations['conv2d_nrn']
    neurons.extra_global_params['Vthr'].view[:] = y.max()
    tg_model.set_inputs(x)


    #############

    tg_model.step_time()

    tg_model.g_model.pull_var_from_device('conv2d_nrn', 'Vmem_peak')
    Vmem_peak = neurons.vars['Vmem_peak'].view.reshape(y.shape)
    print(Vmem_peak)


    tg_model.step_time()

    tg_model.g_model.pull_var_from_device('conv2d_nrn', 'Vmem_peak')
    Vmem_peak = neurons.vars['Vmem_peak'].view.reshape(y.shape)
    print(Vmem_peak)




    print(Vmem_peak == y)
    print((Vmem_peak == y).all())
    return



    print('###########################')


    # IF_INPUT, POISSON_INPUT, SPIKE_INPUT

    # IF_INPUT IS SEPARATE MERGED IF AND CS MODEL
    # POISSON_INPUT ALREADY DONE
    # SPIKE_INPUT IS SIMPLE 1 BOOLEAN VARIABLE MODEL - IF VAR==TRUE IN THRESHOLD CODE



    #############


    #tg_model.step_time(iterations=y.max().astype(np.uint32))


    tg_model.g_model.pull_var_from_device(neurons.name, 'nSpk')
    tg_y = neurons.vars['nSpk'].view.reshape(y.shape).astype(dtype=np.float32)

    #print(tg_y)
    print(y)

    print(tg_y == y)
    print((tg_y == y).all())



if __name__ == '__main__':
    test_conv2d_1_in_chan_1_out_chan_1_stride_valid()