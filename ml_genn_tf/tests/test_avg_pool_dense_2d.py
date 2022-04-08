import numpy as np
import tensorflow as tf
import ml_genn as mlg
import pytest
from converter import Converter

@pytest.mark.parametrize(
    "in_size, in_chan, out_size, pool_size, pool_strides, prefer_in_memory_connect", 
    [(20, 1, 10, 2, 2, True),
     (20, 1, 10, 2, 2, False),
     (20, 2, 10, 2, 2, True),
     (20, 2, 10, 2, 2, False),
     (20, 1, 10, 3, 3, True),
     (20, 1, 10, 3, 3, False),
     (20, 1, 10, 2, 3, True),
     (20, 1, 10, 2, 3, False)])
def test_avg_pool_dense_2d(in_size, in_chan, out_size, pool_size,
                           pool_strides, prefer_in_memory_connect, request):
    # Don"t use all GPU memory for TF!
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Generate input tensor
    x = np.random.randint(0, 2, size=(1, in_size, in_size, in_chan)).astype(np.float64)

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(
            pool_size, strides=pool_strides, padding="valid",
            input_shape=(in_size, in_size, in_chan)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(out_size, use_bias=False)])

    # Generate and set weights
    w = tf_model.get_weights()[0]
    w[:] = np.random.random_sample(w.shape)
    tf_model.set_weights([w])

    # Run TensorFlow model
    tf_y = tf_model([x]).numpy()

    # Convert and compile ML GeNN model
    converter = Converter()
    mlg_net, mlg_net_inputs, mlg_net_outputs = converter.convert(tf_model)
    
    compiler = converter.create_compiler(prefer_in_memory_connect=prefer_in_memory_connect)
    compiled_net = compiler.compile(mlg_net, request.keywords.node.name, 
                                    inputs=mlg_net_inputs, 
                                    outputs=mlg_net_outputs)

    with compiled_net:
        # Evaluate ML GeNN model
        accuracy = compiled_net.evaluate_numpy({mlg_net_inputs[0]: x},
                                               {mlg_net_outputs[0]: tf_y},
                                               "mean_square_error")
    assert accuracy[mlg_net_outputs[0]].result < 1e-03