import tensorflow as tf

from ml_genn.layers import InputType
from ml_genn.layers import IFNeurons
from ml_genn.layers import SpikeInputNeurons
from ml_genn.layers import PoissonInputNeurons
from ml_genn.layers import IFInputNeurons

class RateBased(object):
    def __init__(self, input_type='poisson'):
        self.input_type = InputType(input_type)

    def validate_tf_layer(self, tf_layer):
        if tf_layer.activation != tf.keras.activations.relu:
            raise NotImplementedError('{} activation not supported'.format(type(tf_layer.activation)))
        if tf_layer.use_bias == True:
            raise NotImplementedError('bias tensors not supported')

    def create_input_neurons(self):
        if self.input_type == InputType.SPIKE:
            return SpikeInputNeurons()
        elif self.input_type == InputType.SPIKE_SIGNED:
            return SpikeInputNeurons(signed_spikes=True)
        elif self.input_type == InputType.POISSON:
            return PoissonInputNeurons()
        elif self.input_type == InputType.POISSON_SIGNED:
            return PoissonInputNeurons(signed_spikes=True)
        elif self.input_type == InputType.IF:
            return IFInputNeurons()

    def create_neurons(self, tf_layer):
        return IFNeurons(threshold=1.0)
