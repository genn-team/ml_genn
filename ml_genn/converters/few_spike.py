import tensorflow as tf

from ml_genn.layers import FSReluNeurons
from ml_genn.layers import FSReluInputNeurons

class FewSpike(object):
    def __init__(self, K=10, alpha=25, signed_input=False):
        self.K = K
        self.alpha = alpha
        self.signed_input = signed_input

    def validate_tf_layer(self, tf_layer):
        if tf_layer.activation != tf.keras.activations.relu:
            raise NotImplementedError('{} activation not supported'.format(type(tf_layer.activation)))
        if tf_layer.use_bias == True:
            raise NotImplementedError('bias tensors not supported')

    def create_input_neurons(self):
        return FSReluInputNeurons(self.K, self.alpha, self.signed_input)

    def create_neurons(self, tf_layer):
        return FSReluNeurons(self.K, self.alpha)
