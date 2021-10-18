import numpy as np
import sys
import tensorflow as tf
from copy import copy
from collections import namedtuple
from six import iteritems

from ml_genn.layers import FSReluNeurons
from ml_genn.layers import FSReluInputNeurons

# Because we want the converter class to be reusable, we don't want the
# normalisation data to be a member, instead we encapsulate it in a tuple
PreCompileOutput = namedtuple('PreCompileOutput', ['layer_alpha', 'input_alpha'])

class FewSpike(object):
    def __init__(self, K=10, alpha=25, signed_input=False, norm_data=None):
        self.K = K
        self.alpha = alpha
        self.signed_input = signed_input
        self.norm_data = norm_data

    def validate_tf_layer(self, tf_layer, config):
        if isinstance(tf_layer, (
                tf.keras.layers.Dense,
                tf.keras.layers.Conv2D)):

            if tf_layer.use_bias:
                # no bias tensors allowed
                raise NotImplementedError('Few-Spike converter: bias tensors not supported')

            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Few-Spike converter: output layer must have ReLU or softmax activation')

            elif config.has_activation:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Few-Spike converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.Activation):
            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Few-Spike converter: output layer must have ReLU or softmax activation')

            else:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Few-Spike converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.ReLU):
            # ReLU activation allowed everywhere
            pass

        elif isinstance(tf_layer, tf.keras.layers.Softmax):
            # softmax activation only allowed for output layers
            if not config.is_output:
                raise NotImplementedError(
                    'Few-Spike converter: only output layers may use softmax')

        elif isinstance(tf_layer, (
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.GlobalAveragePooling2D)):
            # average pooling allowed
            pass

        else:
            # no other layers allowed
            raise NotImplementedError(
                'Few-Spike converter: {} layers are not supported'.format(
                    tf_layer.__class__.__name__))

    def create_input_neurons(self, pre_convert_output):
        alpha = (self.alpha if pre_convert_output.input_alpha is None 
                 else pre_convert_output.input_alpha)
        return FSReluInputNeurons(self.K, alpha, self.signed_input)

    def create_neurons(self, tf_layer, pre_convert_output):
        # Lookup optimised alpha value for neuron
        alpha = (pre_convert_output.layer_alpha[tf_layer]
                 if tf_layer in pre_convert_output.layer_alpha 
                 else self.alpha)
        return FSReluNeurons(self.K, alpha)
    
    def pre_convert(self, tf_model):
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
            layer_alpha = {l: np.max(out) / (1.0 - 2.0 ** -self.K)
                           for l, out in zip(weighted_layers, outputs)}

            # Use input data range to directly set maximum input
            if self.signed_input:
                input_alpha = np.amax(np.abs(self.norm_data)) / (1.0 - 2.0 ** (1 - self.K))
            else:
                input_alpha = np.amax(self.norm_data) / (1.0 - 2.0 ** -self.K)

            # Return results of normalisation in tuple
            return PreCompileOutput(layer_alpha=layer_alpha,
                                    input_alpha=input_alpha)

        # Otherwise, return empty normalisation output tuple
        else:
            return PreCompileOutput(layer_alpha={}, input_alpha=None)

    def pre_compile(self, mlg_model):
        delay_to_layers = {}
        
        # Loop through layers
        for l in mlg_model.layers:
            # If layer has upstream synaptic connections
            if len(l.upstream_synapses) > 0:
                # Calculate max delay from upstream synapses
                max_delay = max(delay_to_layers[s.source()]
                                for s in l.upstream_synapses)
                
                # Delay to this layer is one more than this
                delay_to_layers[l] = 1 + max_delay
                
                # Determine the maximum alpha value upstream layers
                max_alpha = max(s.source().neurons.alpha
                                for s in l.upstream_synapses)

                # Loop through upstream synapses
                for s in l.upstream_synapses:
                    # Set delay to balance
                    s.delay = (max_delay - delay_to_layers[s.source()]) * self.K

                    # Set alpha to maximum
                    s.source().neurons.alpha = max_alpha

            # Otherwise (layer is an input layer), set this layer's delay as zero
            else:
                delay_to_layers[l] = 0
    

    def post_compile(self, mlg_model):
        # do not allow multiple input or output layers
        if len(mlg_model.inputs) > 1 or len(mlg_model.outputs) > 1:
            raise NotImplementedError(
                'multiple input or output layers not supported for Few Spike conversion')
