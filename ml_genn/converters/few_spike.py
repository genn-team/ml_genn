import numpy as np
import tensorflow as tf

from ml_genn.layers import FSReluNeurons
from ml_genn.layers import FSReluInputNeurons

class FewSpike(object):
    def __init__(self, K=10, alpha=25, signed_input=False):
        self.K = K
        self.alpha = alpha
        self.signed_input = signed_input
        self.max_activations = {}
        
    def optimise_alpha(self, tf_model, norm_data):
        # Get weighted layers
        weighted_layers = [l for l in tf_model.layers
                           if len(l.get_weights()) > 0]
        
        # Get output functions for weighted layers.
        get_outputs = tf.keras.backend.function(
            tf_model.inputs, [l.output for l in weighted_layers])
            
        # Get output given input data.
        outputs = get_outputs(norm_data)
        
        # Build dictionary of maximum activation in each layer
        self.max_activations = {l: np.max(out)
                                for l, out in zip(weighted_layers, outputs)}

    def validate_tf_layer(self, tf_layer):
        if tf_layer.activation != tf.keras.activations.relu:
            raise NotImplementedError('{} activation not supported'.format(type(tf_layer.activation)))
        if tf_layer.use_bias == True:
            raise NotImplementedError('bias tensors not supported')

    def create_input_neurons(self):
        return FSReluInputNeurons(self.K, self.alpha, self.signed_input)

    def create_neurons(self, tf_layer):
        # Lookup optimised alpha value for neuron
        alpha = (float(np.ceil(self.max_activations[tf_layer]))
                 if tf_layer in self.max_activations 
                 else self.alpha)
        return FSReluNeurons(self.K, alpha)
