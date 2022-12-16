import numpy as np
import tensorflow as tf
import pytest

from .converter import Converter
from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, BinarySpikeInput
from ml_genn.connectivity import Conv2DTranspose

@pytest.mark.parametrize(
    "in_size, in_chan, out_chan, conv_size, conv_strides, conv_padding, prefer_in_memory_connect", 
    [(12, 1, 1, 3, 1, "valid", True),
     (12, 1, 1, 3, 1, "valid", False),
     (12, 1, 1, 3, 1, "same", True),
     (12, 1, 1, 3, 1, "same", False),
     (12, 2, 1, 3, 1, "valid", True),
     (12, 2, 1, 3, 1, "valid", False),
     (12, 1, 2, 3, 1, "valid", True),
     (12, 1, 2, 3, 1, "valid", False),
     (12, 2, 2, 3, 1, "valid", True),
     (12, 2, 2, 3, 1, "valid", False),
     (12, 1, 1, 4, 1, "valid", True),
     (12, 1, 1, 4, 1, "valid", False),
     (12, 1, 1, 4, 1, "same", True),
     (12, 1, 1, 4, 1, "same", False),
     (12, 1, 1, 3, 2, "valid", True),
     (12, 1, 1, 3, 2, "valid", False),
     (12, 1, 1, 3, 2, "same", True),
     (12, 1, 1, 3, 2, "same", False)])
def test_conv_2d_transpose(in_size, in_chan, out_chan, conv_size, 
                           conv_strides, conv_padding, 
                           prefer_in_memory_connect, request):
    # Don't use all GPU memory for TF!
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Generate input tensor
    x = np.random.randint(0, 2, size=(1, in_size, in_size, in_chan)).astype(np.float64)  

    # Create TensorFlow model
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2DTranspose(
            out_chan, conv_size, strides=conv_strides, padding=conv_padding,
            use_bias=False, input_shape=(in_size, in_size, in_chan))])

    # Generate and set weights
    w = np.random.random_sample((conv_size, conv_size, out_chan, in_chan))
    tf_model.set_weights([w])

    # Run TensorFlow model
    tf_y = tf_model([x]).numpy()
    
    # Create sequential model
    network = SequentialNetwork()
    with network:
        input = InputLayer(BinarySpikeInput(),  (in_size, in_size, in_chan))
        output = Layer(Conv2DTranspose(w, out_chan, conv_size, conv_strides=conv_strides, 
                                       conv_padding=conv_padding), 
                       IntegrateFire(v_thresh=np.float64(np.finfo(np.float32).max), readout="var"))

    compiler = InferenceCompiler(evaluate_timesteps=2)
    compiled_net = compiler.compile(network, request.keywords.node.name)

    with compiled_net:
        # Evaluate ML GeNN model
        metrics, _ = compiled_net.evaluate({input: x},
                                           {output: tf_y},
                                           "mean_square_error")
    assert metrics[output].result < 1e-03
