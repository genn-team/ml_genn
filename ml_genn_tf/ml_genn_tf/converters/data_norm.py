import logging
import numpy as np
import tensorflow as tf

from collections import namedtuple
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import (BinarySpikeInput, IntegrateFire,
                             IntegrateFireInput, Neuron, PoissonInput)
from .converter import Converter
from .enum import InputType

logger = logging.getLogger(__name__)

# Because we want the converter class to be reusable, we don't want the
# normalisation data to be a member, instead we encapsulate it in a tuple
PreConvertOutput = namedtuple("PreConvertOutput", ["thresholds"])

class DataNorm(Converter):
    """Converts ANNs to network of integrate-and-fire neurons, 
    operating in a rate-based regime If normalisation data is provided,
    thresholds are balancing using the algorithm proposed by [Diehl2015]_.
    
    Args:
        evaluate_timesteps: ss
        signed_input:       ss
        norm_data:          paa
        input_type:         sss
    """
    def __init__(self, evaluate_timesteps: int, signed_input=False,
                 norm_data=None, input_type=InputType.POISSON):
        self.norm_data = norm_data
        self.evaluate_timesteps = evaluate_timesteps
        self.signed_input = signed_input
        self.input_type = InputType(input_type)

    def create_input_neurons(self,
                             pre_convert_output: PreConvertOutput) -> Neuron:
        if self.input_type == InputType.SPIKE:
            return BinarySpikeInput(signed_spikes=self.signed_input)
        elif self.input_type == InputType.POISSON:
            return PoissonInput(signed_spikes=self.signed_input)
        elif self.input_type == InputType.IF:
            return IntegrateFireInput()

    def create_neurons(self, tf_layer: tf.keras.layers.Layer,
                       pre_convert_output: PreConvertOutput, 
                       is_output: bool) -> Neuron:
        threshold = pre_convert_output.thresholds[tf_layer]
        logger.debug(f"layer {tf_layer.name}: threshold={threshold}")
              
        return IntegrateFire(v_thresh=threshold,
                             readout="spike_count" if is_output else None)

    def pre_convert(self, tf_model):
        # NOTE: Data-Norm only normalises an initial sequential portion of
        # a model with one input layer. Models with multiple input layers
        # are not currently supported, and the thresholds of layers after
        # branching (e.g. in ResNet) are left at 1.0.

        # Default to 1.0 threshold
        scale_factors = {l: np.float64(1.0) for l in tf_model.layers}
        thresholds = {l: np.float64(1.0) for l in tf_model.layers}

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
            tf_out_layers = [n.outbound_layer 
                             for n in tf_layer.outbound_nodes
                             if n in tf_model_nodes]
            tf_out_layers_all[tf_layer] = tf_out_layers

        # Get input layers
        if isinstance(tf_model, tf.keras.models.Sequential):
            # In TF Sequential models, the InputLayer is not stored in the 
            # model object, so we must traverse back through nodes to find 
            # the input layer's outputs.
            tf_in_layers = tf_in_layers_all[tf_model.layers[0]]
            assert(len(tf_in_layers) == 1)
            tf_out_layers = [n.outbound_layer 
                             for n in tf_in_layers[0].outbound_nodes
                             if n in tf_model_nodes]
            scale_factors[tf_in_layers[0]] = np.float64(1.0)
            thresholds[tf_in_layers[0]] = np.float64(1.0)
            tf_in_layers_all[tf_in_layers[0]] = []
            tf_out_layers_all[tf_in_layers[0]] = tf_out_layers

        else:
            # TF Functional models store all their InputLayers, 
            # so no trickery needed.
            tf_in_layers = [tf_model.get_layer(name) 
                            for name in tf_model.input_names]

        for tf_in_layer in tf_in_layers:
            assert(len(tf_in_layer.output_shape) == 1)

            # input layers cannot be output layers
            if len(tf_out_layers_all[tf_in_layer]) == 0:
                raise NotImplementedError("input layers as output layers not "
                                          "supported")

        # Don't allow models with multiple input layers
        if len(tf_in_layers) != 1:
            raise NotImplementedError("Data-Norm converter: models with "
                                      "multiple input layers not supported")

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
                layer_out_fn = tf.keras.backend.function(tf_model.inputs,
                                                         tf_layer.output)

                max_activation = 0
                for d, _ in norm_data_iter:
                    # Get max output given norm data batch.
                    activation = layer_out_fn(d)
                    max_activation = np.maximum(max_activation, 
                                                np.max(activation))

                max_weight = np.max(tf_layer.get_weights()[0])
                scale_factor = np.maximum(max_activation, max_weight)
                threshold = scale_factor / scale_factors[tf_in_layers[0]]
                logger.debug(f"layer {tf_layer.name}:"
                             f"max activation={max_activation}, "
                             f"max weight={max_weight}")

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

        return PreConvertOutput(thresholds=thresholds)

    def create_compiler(self, **kwargs):
        return InferenceCompiler(evaluate_timesteps=self.evaluate_timesteps,
                                 **kwargs)