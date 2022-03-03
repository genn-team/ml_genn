import numpy as np
import tensorflow as tf
import ml_genn as mlg
import pytest

@pytest.mark.parametrize(
    'in_size, which_on', [
        (5, 'all'),
        (5, 'none'),
        (5, 'random'),
        (10, 'random'),
    ])

def test_identity(in_size, which_on, request):
    # Don't use all GPU memory for TF!
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Generate input tensor
    if which_on == 'all':
        x = np.ones((1, in_size), dtype=np.float64)
    elif which_on == 'none':
        x = np.zeros((1, in_size), dtype=np.float64)
    else: # 'random'
        x = np.random.randint(0, 2, size=(1, in_size)).astype(np.float64)

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            in_size, use_bias=False, input_shape=(in_size,)),
    ], name=request.keywords.node.name)

    # Generate and set weights
    w = np.identity(in_size, dtype=np.float64)
    tf_model.set_weights([w])

    # Run TensorFlow model
    tf_y = tf_model([x]).numpy()

    # Run ML GeNN model
    mlg_input = mlg.layers.InputLayer('input', (in_size,), neurons=mlg.layers.SpikeInputNeurons())
    mlg_output = mlg.layers.Identity('identity')
    mlg_output.connect([mlg_input])

    mlg_model = mlg.Model([mlg_input], [mlg_output], name=request.keywords.node.name)
    mlg_model.compile()
    mlg_model.outputs[0].neurons.set_threshold(np.float64(np.inf))
    mlg_model.set_input_batch([x])
    mlg_model.step_time(2)

    nrn = mlg_model.outputs[0].neurons.nrn
    nrn.pull_var_from_device('Vmem')
    mlg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert(np.allclose(mlg_y, tf_y, atol=0.0, rtol=1.0e-3))
