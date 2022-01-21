import numpy as np
import tensorflow as tf
import ml_genn as mlg
import pytest
from converter import Converter

@pytest.mark.parametrize(
    'in_size, out_size, which_on', [
        (5, 5, 'all'),
        (5, 5, 'none'),
        (5, 5, 'random'),
        (5, 10, 'random'),
    ])

def test_dense(in_size, out_size, which_on, request):
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

    # Generate random weights
    w = np.random.random_sample((in_size, out_size))

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(out_size, name='output', use_bias=False, input_shape=(in_size,)),
    ], name=request.keywords.node.name)
    tf_model.set_weights([w])

    # Run TensorFlow model
    tf_y = tf_model([x]).numpy()

    # Run ML GeNN model
    mlg_model = mlg.Model.convert_tf_model(tf_model, converter=Converter())
    mlg_model.outputs[0].neurons.set_threshold(np.float64(np.inf))
    mlg_model.set_input_batch([x])
    mlg_model.step_time(2)

    nrn = mlg_model.outputs[0].neurons.nrn
    nrn.pull_var_from_device('Vmem')
    mlg_y = nrn.vars['Vmem'].view.reshape(tf_y.shape)

    assert(np.allclose(mlg_y, tf_y, atol=0.0, rtol=1.0e-3))
