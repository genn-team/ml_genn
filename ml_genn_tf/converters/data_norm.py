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

        elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
            # global average pooling allowed
            pass
        elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
            if tf_layer.padding != 'valid':
                raise NotImplementedError(
                    'Data-Norm converter: only valid padding is supported for pooling layers')

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
        # NOTE: Data-Norm only normalises an initial sequential portion of
        # a model with one input layer. Models with multiple input layers
        # are not currently supported, and the thresholds of layers after
        # branching (e.g. in ResNet) are left at 1.0.

        # Default to 1.0 threshold
        scale_factors = {tf_layer: np.float64(1.0) for tf_layer in tf_model.layers}
        thresholds = {tf_layer: np.float64(1.0) for tf_layer in tf_model.layers}

        # Only traverse nodes belonging to this model
        tf_model_nodes = set()
        for n in tf_model._nodes_by_depth.values():
            tf_model_nodes.update(n)

        # Get inbound and outbound layers
        tf_in_layers_all = {}
        tf_out_layers_all = {}
        for tf_layer in tf_model.layers:

            # Find inbound layers
            tf_in_layers = []
            for n in tf_layer.inbound_nodes:
                if n not in tf_model_nodes:
                    continue
                if isinstance(n.inbound_layers, list):
                    tf_in_layers += n.inbound_layers
                else:
                    tf_in_layers.append(n.inbound_layers)
            tf_in_layers_all[tf_layer] = tf_in_layers

            # Find outbound layers
            tf_out_layers = [n.outbound_layer for n in tf_layer.outbound_nodes
                             if n in tf_model_nodes]
            tf_out_layers_all[tf_layer] = tf_out_layers

        # Get input layers
        if isinstance(tf_model, tf.keras.models.Sequential):
            # In TF Sequential models, the InputLayer is not stored in the model object,
            # so we must traverse back through nodes to find the input layer's outputs.
            tf_in_layers = tf_in_layers_all[tf_model.layers[0]]
            assert(len(tf_in_layers) == 1)
            tf_out_layers = [n.outbound_layer for n in tf_in_layers[0].outbound_nodes
                             if n in tf_model_nodes]
            scale_factors[tf_in_layers[0]] = np.float64(1.0)
            thresholds[tf_in_layers[0]] = np.float64(1.0)
            tf_in_layers_all[tf_in_layers[0]] = []
            tf_out_layers_all[tf_in_layers[0]] = tf_out_layers

        else:
            # TF Functional models store all their InputLayers, so no trickery needed.
            tf_in_layers = [tf_model.get_layer(name) for name in tf_model.input_names]

        for tf_in_layer in tf_in_layers:
            assert(len(tf_in_layer.output_shape) == 1)

            # input layers cannot be output layers
            if len(tf_out_layers_all[tf_in_layer]) == 0:
                raise NotImplementedError(
                    'input layers as output layers not supported')

        # Don't allow models with multiple input layers
        if len(tf_in_layers) != 1:
            raise NotImplementedError(
                'Data-Norm converter: models with multiple input layers not supported')

        tf_layer = tf_in_layers[0]

        while True:
            tf_in_layers = tf_in_layers_all[tf_layer]
            tf_out_layers = tf_out_layers_all[tf_layer]

            # Skip input layer
            if len(tf_in_layers) == 0:
                tf_layer = tf_out_layers[0]
                continue

            # Break at branch (many outbound)
            if len(tf_out_layers) > 1:
                break

            # If layer is weighted, compute max activation and weight
            if len(tf_layer.get_weights()) > 0:
                norm_data_iter = iter(self.norm_data[0])

                # Get output function for layer
                layer_out_fn = tf.keras.backend.function(tf_model.inputs, tf_layer.output)

                max_activation = 0
                for d, _ in norm_data_iter:
                    # Get max output given norm data batch.
                    activation = layer_out_fn(d)
                    max_activation = np.maximum(max_activation, np.max(activation))

                max_weight = np.max(tf_layer.get_weights()[0])
                scale_factor = np.maximum(max_activation, max_weight)
                threshold = scale_factor / scale_factors[tf_in_layers[0]]
                print(f'layer <{tf_layer.name}  max activation {max_activation}  max weight {max_weight}')

            else:
                # If layer is not weighted (like ReLU or Flatten),
                # use data from inbound layer (like Dense or Conv2D)
                scale_factor = scale_factors[tf_in_layers[0]]
                threshold = thresholds[tf_in_layers[0]]

            scale_factors[tf_layer] = scale_factor
            thresholds[tf_layer] = threshold

            # Break at end (no outbound)
            if len(tf_out_layers) == 0:
                break

            # Outbound layer
            tf_layer = tf_out_layers[0]

        return PreCompileOutput(thresholds=thresholds)
    
    def pre_compile(self, mlg_model):
        pass

    def post_compile(self, mlg_model):
        # For each layer (these *should* be topologically sorted)
        for layer in mlg_model.layers:
            if layer in mlg_model.inputs:
                continue

            print('layer <{}> threshold: {}'.format(layer.name, layer.neurons.threshold))
