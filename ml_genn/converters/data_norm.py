import tensorflow as tf
import numpy as np
from collections import namedtuple

from ml_genn.layers import InputType
from ml_genn.layers import IFNeurons
from ml_genn.layers import SpikeInputNeurons
from ml_genn.layers import PoissonInputNeurons
from ml_genn.layers import IFInputNeurons

# Because we want the converter class to be reusable, we don't want the
# normalisation data to be a member, instead we encapsulate it in a tuple
PreCompileOutput = namedtuple('PreCompileOutput', ['thresholds'])

class DataNorm(object):
    def __init__(self, norm_data, signed_input=False, 
                 input_type=InputType.POISSON):
        self.norm_data = norm_data
        self.signed_input = signed_input
        self.input_type = InputType(input_type)

    def validate_tf_layer(self, tf_layer, config):
        if isinstance(tf_layer, (
                tf.keras.layers.Dense,
                tf.keras.layers.Conv2D)):

            if tf_layer.use_bias:
                # no bias tensors allowed
                raise NotImplementedError('Data-Norm converter: bias tensors not supported')

            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Data-Norm converter: output layer must have ReLU or softmax activation')

            elif config.has_activation:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Data-Norm converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.Activation):
            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError(
                        'Data-Norm converter: output layer must have ReLU or softmax activation')

            else:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError(
                        'Data-Norm converter: hidden layers must have ReLU activation')

        elif isinstance(tf_layer, tf.keras.layers.ReLU):
            # ReLU activation allowed everywhere
            pass

        elif isinstance(tf_layer, tf.keras.layers.Softmax):
            # softmax activation only allowed for output layers
            if not config.is_output:
                raise NotImplementedError(
                    'Data-Norm converter: only output layers may use softmax')

        elif isinstance(tf_layer, (
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.GlobalAveragePooling2D)):
            # average pooling allowed
            pass

        else:
            # no other layers allowed
            raise NotImplementedError(
                'Data-Norm converter: {} layers are not supported'.format(
                    tf_layer.__class__.__name__))

    def create_input_neurons(self, pre_convert_output):
        if self.input_type == InputType.SPIKE:
            return SpikeInputNeurons(signed_spikes=self.signed_input)
        elif self.input_type == InputType.POISSON:
            return PoissonInputNeurons(signed_spikes=self.signed_input)
        elif self.input_type == InputType.IF:
            return IFInputNeurons()

    def create_neurons(self, tf_layer, pre_convert_output):
        return IFNeurons(threshold=pre_convert_output.thresholds[tf_layer])

    def pre_convert(self, tf_model):
        # Get output functions for weighted layers.
        idx = [i for i, layer in enumerate(tf_model.layers)
               if len(layer.get_weights()) > 0]
        weighted_layers = [tf_model.layers[i] for i in idx]
        get_outputs = tf.keras.backend.function(
            tf_model.inputs, [layer.output for layer in weighted_layers])

        # Find the maximum activation in each layer, given input data.
        max_activation = np.array([np.max(out) for out in get_outputs(self.norm_data)],
                                  dtype=np.float64)

        # Find the maximum weight in each layer.
        max_weights = np.array([np.max(w) for w in tf_model.get_weights()],
                               dtype=np.float64)

        # Compute scale factors and normalize weights.
        scale_factors = np.max([max_activation, max_weights], 0)
        applied_factors = np.empty(scale_factors.shape, dtype=np.float64)
        applied_factors[0] = scale_factors[0]
        applied_factors[1:] = scale_factors[1:] / scale_factors[:-1]

        for layer, threshold in zip(weighted_layers, applied_factors):
            print('layer <{}> threshold: {}'.format(layer.name, threshold))

        # Build dictionary of thresholds for each layer
        thresholds = {layer: threshold for layer, threshold
                      in zip(weighted_layers, applied_factors)}

        return PreCompileOutput(thresholds=thresholds)
    
    def pre_compile(self, mlg_model):
        pass

    def post_compile(self, mlg_model):
        pass
