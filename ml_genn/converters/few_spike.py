import numpy as np
import tensorflow as tf
from collections import namedtuple

from ml_genn.layers import FSReluNeurons
from ml_genn.layers import FSReluInputNeurons

# Because we want the converter class to be reusable, we don't want the
# normalisation data to be a member, instead we encapsulate it in a tuple
PreCompileOutput = namedtuple('PreCompileOutput', ['max_activations', 'max_input'])

class FewSpike(object):
    def __init__(self, K=10, alpha=25, signed_input=False, norm_data=None):
        self.K = K
        self.alpha = alpha
        self.signed_input = signed_input
        self.norm_data = norm_data

    def validate_tf_layer(self, tf_layer):
        if tf_layer.activation != tf.keras.activations.relu:
            raise NotImplementedError('{} activation not supported'.format(type(tf_layer.activation)))
        if tf_layer.use_bias == True:
            raise NotImplementedError('bias tensors not supported')

    def create_input_neurons(self, pre_compile_output):
        alpha = (self.alpha if pre_compile_output.max_input is None 
                 else float(np.ceil(pre_compile_output.max_input)))
        return FSReluInputNeurons(self.K, alpha, self.signed_input)

    def create_neurons(self, tf_layer, pre_compile_output):
        # Lookup optimised alpha value for neuron
        alpha = (float(np.ceil(pre_compile_output.max_activations[tf_layer]))
                 if tf_layer in pre_compile_output.max_activations 
                 else self.alpha)
        return FSReluNeurons(self.K, alpha)
    
    def pre_compile(self, tf_model):
        # If any normalisation data was provided
        if self.norm_data is not None:
            # Get weighted layers
            weighted_layers = [l for l in tf_model.layers
                               if len(l.get_weights()) > 0]

            # Get output functions for weighted layers.
            get_outputs = tf.keras.backend.function(
                tf_model.inputs, [l.output for l in weighted_layers])

            # Get output given input data.
            outputs = get_outputs(self.norm_data)

            # Build dictionary of maximum activation in each layer
            max_activations = {l: np.max(out)
                               for l, out in zip(weighted_layers, outputs)}

            # Use input data range to directly set maximum input
            if self.signed_input:
                max_input = np.amax(np.abs(self.norm_data))
            else:
                max_input = np.amax(self.norm_data)

            # Return results of normalisation in tuple
            return PreCompileOutput(max_activations=max_activations,
                                    max_input=max_input)

        # Otherwise, return empty normalisation output tuple
        else:
            return PreCompileOutput(max_activations={}, max_input=None)
    
    def post_compile(self, mlg_model):
        pass
