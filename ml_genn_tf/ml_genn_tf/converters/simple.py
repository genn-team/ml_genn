import tensorflow as tf

from ml_genn.neurons import (BinarySpikeInput, IntegrateFire,
                             IntegrateFireInput, PoissonInput)
from .enum import InputType

class Simple(object):
    def __init__(self, signed_input:bool=False, input_type:InputType=InputType.POISSON):
        self.signed_input = signed_input
        self.input_type = InputType(input_type)

    def validate_tf_layer(self, tf_layer, config):
        if isinstance(tf_layer, (tf.keras.layers.Dense, 
                                 tf.keras.layers.Conv2D)):
            if tf_layer.use_bias:
                # no bias tensors allowed
                raise NotImplementedError('Simple converter: bias tensors not supported')

            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Simple converter: output layer must have ReLU or softmax activation')

            elif config.has_activation:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Simple converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.Activation):
            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Simple converter: output layer must have ReLU or softmax activation')

            else:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Simple converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.ReLU):
            # ReLU activation allowed everywhere
            pass

        elif isinstance(tf_layer, tf.keras.layers.Softmax):
            # softmax activation only allowed for output layers
            if not config.is_output:
                raise NotImplementedError(
                    'Simple converter: only output layers may use softmax')

        elif isinstance(tf_layer, (tf.keras.layers.AveragePooling2D,
                                   tf.keras.layers.GlobalAveragePooling2D)):
            # average pooling allowed
            pass

        else:
            # no other layers allowed
            raise NotImplementedError(
                'Simple converter: {} layers are not supported'.format(
                    tf_layer.__class__.__name__))

    def create_input_neurons(self, pre_compile_output):
        if self.input_type == InputType.SPIKE:
            return BinarySpikeInput(signed_spikes=self.signed_input)
        elif self.input_type == InputType.POISSON:
            return PoissonInput(signed_spikes=self.signed_input)
        elif self.input_type == InputType.IF:
            return IntegrateFireInput()

    def create_neurons(self, tf_layer, pre_compile_output, is_output):
        return IntegrateFire(v_thresh=1.0, 
                             output="spike_count" if is_output else None)

    def pre_convert(self, tf_model):
        pass
    
    def pre_compile(self, mlg_network):
        pass