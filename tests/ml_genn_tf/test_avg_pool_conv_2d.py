import numpy as np
import tensorflow as tf
import pytest
from .converter import Converter

@pytest.mark.parametrize(
    "in_size, in_chan, out_chan, pool_size, pool_strides, conv_size, conv_strides, conv_padding, prefer_in_memory_connect", 
    [(20, 1, 1, 2, 2, 3, 1, "valid", True),
     (20, 1, 1, 2, 2, 3, 1, "valid", False),
     (20, 2, 2, 2, 2, 3, 1, "valid", True),
     (20, 2, 2, 2, 2, 3, 1, "valid", False),
     (20, 1, 1, 2, 2, 3, 1, "same", True),
     (20, 1, 1, 2, 2, 3, 1, "same", False),
     (20, 2, 2, 2, 2, 3, 1, "same", True),
     (20, 2, 2, 2, 2, 3, 1, "same", False),
     (20, 1, 1, 3, 3, 4, 1, "valid", True),
     (20, 1, 1, 3, 3, 4, 1, "valid", False),
     (20, 1, 1, 3, 4, 4, 1, "valid", True),
     (20, 1, 1, 3, 4, 4, 1, "valid", False),
     (20, 1, 1, 3, 3, 4, 1, "same", True),
     (20, 1, 1, 3, 3, 4, 1, "same", False),
     (20, 1, 1, 3, 4, 4, 1, "same", True),
     (20, 1, 1, 3, 4, 4, 1, "same", False)])
def test_avg_pool_conv_2d(in_size, in_chan, out_chan, pool_size, pool_strides,
                          conv_size, conv_strides, conv_padding, 
                          prefer_in_memory_connect, request):
    # Don't use all GPU memory for TF!
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Generate input tensor
    x = np.random.randint(0, 2, size=(1, in_size, in_size, in_chan)).astype(np.float64)

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.AveragePooling2D(
            pool_size, strides=pool_strides, padding="valid",
            input_shape=(in_size, in_size, in_chan)),
        tf.keras.layers.Conv2D(
            out_chan, conv_size, strides=conv_strides, padding=conv_padding, use_bias=False)])

    # Generate and set weights
    w = np.random.random_sample((conv_size, conv_size, in_chan, out_chan))
    tf_model.set_weights([w])

    # Run TensorFlow model
    tf_y = tf_model([x]).numpy()

    # Convert and compile ML GeNN model
    converter = Converter()
    net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)
    
    compiler = converter.create_compiler(prefer_in_memory_connect=prefer_in_memory_connect)
    compiled_net = compiler.compile(net, request.keywords.node.name, 
                                    inputs=net_inputs, 
                                    outputs=net_outputs)

    with compiled_net:
        # Evaluate ML GeNN model
        metrics, _ = compiled_net.evaluate({net_inputs[0]: x},
                                           {net_outputs[0]: tf_y},
                                           "mean_square_error")
    assert metrics[net_outputs[0]].result < 1e-03
