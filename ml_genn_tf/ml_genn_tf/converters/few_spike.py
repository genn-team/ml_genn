import logging
import numpy as np
import tensorflow as tf

from collections import namedtuple
from ml_genn.compilers import FewSpikeCompiler
from ml_genn.neurons import FewSpikeRelu, FewSpikeReluInput
from .converter import Converter

from copy import copy

logger = logging.getLogger(__name__)

# Because we want the converter class to be reusable, we don't want the
# normalisation data to be a member, instead we encapsulate it in a tuple
PreConvertOutput = namedtuple("PreConvertOutput", ["layer_alpha", "input_alpha"])

class FewSpike(Converter):
    def __init__(self, k: int=10, alpha: float=25, signed_input=False, norm_data=None):
        self.k = k
        self.alpha = alpha
        self.signed_input = signed_input
        self.norm_data = norm_data

    def create_input_neurons(self, pre_convert_output):
        alpha = (self.alpha if pre_convert_output.input_alpha is None 
                 else pre_convert_output.input_alpha)
        return FewSpikeReluInput(self.k, alpha, self.signed_input)

    def create_neurons(self, tf_layer, pre_convert_output, is_output):
        # If layer alphas have been calculated but
        # this layer isn't included, give warning
        pre_conv_alpha = pre_convert_output.layer_alpha
        if len(pre_conv_alpha) > 0 and tf_layer not in pre_conv_alpha:
            logger.warning(f"FewSpike pre_convert has not provided "
                           f"an alpha value for '{tf_layer.name}'")

        # Lookup optimised alpha value for neuron
        alpha = (pre_conv_alpha[tf_layer] if tf_layer in pre_conv_alpha
                 else self.alpha)
        return FewSpikeRelu(self.k, alpha, 
                            readout="var" if is_output else None)

    def pre_convert(self, tf_model):
        # If any normalisation data was provided
        if self.norm_data is not None:
            norm_data_iter = iter(self.norm_data[0])

            # Get output functions for all layers.
            get_outputs = tf.keras.backend.function(
                tf_model.inputs, [l.output for l in tf_model.layers])

            layer_alpha = {l: 0.0 for l in tf_model.layers}
            input_alpha = 0.0
            for d, _ in norm_data_iter:
                # Get output given input data.
                output = get_outputs(d)
                for l, out in zip(tf_model.layers, output):
                    layer_alpha[l] = max(layer_alpha[l], np.amax(out) / (1.0 - 2.0 ** -self.k))

                # Use input data range to directly set maximum input
                if self.signed_input:
                    input_alpha = max(input_alpha, np.amax(np.abs(d)) / (1.0 - 2.0 ** (1 - self.k)))
                else:
                    input_alpha = max(input_alpha, np.amax(d) / (1.0 - 2.0 ** -self.k))

            # Return results of normalisation in tuple
            return PreConvertOutput(layer_alpha=layer_alpha,
                                    input_alpha=input_alpha)

        # Otherwise, return empty normalisation output tuple
        else:
            return PreConvertOutput(layer_alpha={}, input_alpha=None)

    def post_convert(self, mlg_network, mlg_network_inputs, mlg_model_outputs):
        # Loop through populations
        for p in mlg_network.populations:
            # If population has incoming connections
            if len(p.incoming_connections) > 0:
                # Determine the maximum alpha value of presynaptic populations
                max_presyn_alpha = max(c().source().neuron.alpha 
                                       for c in p.incoming_connections)

                # Loop through incoming connections
                for c in p.incoming_connections:
                    # Set presyn alpha to maximum alpha of all presyn layers
                    c().source().neuron.alpha = max_presyn_alpha

    def create_compiler(self, **kwargs):
        return FewSpikeCompiler(k=self.k, **kwargs)
