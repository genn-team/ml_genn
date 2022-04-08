import logging
import tensorflow as tf

from collections import namedtuple
from ml_genn import Connection, Network, Population
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import (AvgPool2D, AvgPoolDense2D, AvgPoolConv2D, 
                                  Conv2D, Dense, OneToOne)

logger = logging.getLogger(__name__)

class Converter:
    def convert(self, tf_model):
        """Create a ML GeNN model from a TensorFlow model

        Args:
        tf_model  --  TensorFlow model to be converted
        """

        tf_activation_layers = (
            tf.keras.layers.Activation,
            tf.keras.layers.ReLU,
            tf.keras.layers.Softmax)

        tf_ignored_layers = (
            tf.keras.layers.Flatten,
            tf.keras.layers.Dropout)

        # only traverse nodes belonging to this model
        tf_model_nodes = set()
        for n in tf_model._nodes_by_depth.values():
            tf_model_nodes.update(n)

        # get inbound and outbound layers
        tf_in_layers_all = {}
        tf_out_layers_all = {}
        for tf_layer in tf_model.layers:

            # find inbound layers
            tf_in_layers = []
            for n in tf_layer.inbound_nodes:
                if n not in tf_model_nodes:
                    continue
                if isinstance(n.inbound_layers, list):
                    tf_in_layers += n.inbound_layers
                else:
                    tf_in_layers.append(n.inbound_layers)
            tf_in_layers_all[tf_layer] = tf_in_layers

            # find outbound layers
            tf_out_layers = [n.outbound_layer for n in tf_layer.outbound_nodes
                             if n in tf_model_nodes]
            tf_out_layers_all[tf_layer] = tf_out_layers


        # Perform any pre-conversion tasks
        pre_convert_output = self.pre_convert(tf_model)

        # configure model build process
        class LayerConfig(object):
            def __init__(self, name, shape, is_input=False, is_output=False,
                         has_activation=False, neurons=None):
                self.name = name
                self.shape = shape
                self.is_input = is_input
                self.is_output = is_output
                self.has_activation = has_activation
                self.neurons = neurons
                self.synapses = []

        InSynConfig = namedtuple('InSynconfig', ['type', 'params', 'source'])

        config_steps = []
        configs_lookups = {}
        new_tf_layers = set()
        traversed_tf_layers = set()

        # get and check input layers
        if isinstance(tf_model, tf.keras.models.Sequential):
            # In TF Sequential models, the InputLayer is not
            # stored in the model object, so we must traverse back
            # through nodes to find the input layer's outputs.
            tf_in_layers = tf_in_layers_all[tf_model.layers[0]]
            assert(len(tf_in_layers) == 1)
            tf_out_layers = [n.outbound_layer 
                             for n in tf_in_layers[0].outbound_nodes
                             if n in tf_model_nodes]
            tf_in_layers_all[tf_in_layers[0]] = []
            tf_out_layers_all[tf_in_layers[0]] = tf_out_layers

        else:
            # TF Functional models store all their  
            # InputLayers, so no trickery needed.
            tf_in_layers = [tf_model.get_layer(name) 
                            for name in tf_model.input_names]

        for tf_in_layer in tf_in_layers:
            assert(len(tf_in_layer.output_shape) == 1)

            # input layers cannot be output layers
            if len(tf_out_layers_all[tf_in_layer]) == 0:
                raise NotImplementedError(
                    'input layers as output layers not supported')


        # === Input Layers ===
        for tf_layer in tf_in_layers:
            new_tf_layers.add(tf_layer)
            logger.debug(f"Configuring Input layer <{tf_layer.name}>")

            # configure layer
            config = LayerConfig(
                tf_layer.name, tf_layer.output_shape[0][1:],
                is_input=True, has_activation=True,
                neurons=self.create_input_neurons(pre_convert_output))

            config_steps.append(config)
            configs_lookups[tf_layer] = [config]


        # while there are still layers to traverse
        while new_tf_layers:
            new_tf_layer = new_tf_layers.pop()
            new_tf_out_layers = tf_out_layers_all[new_tf_layer]
            traversed_tf_layers.add(new_tf_layer)

            # get next TF layer to configure
            for tf_layer in new_tf_out_layers:
                tf_in_layers = tf_in_layers_all[tf_layer]
                tf_out_layers = tf_out_layers_all[tf_layer]

                # skip if we still need to configure inbound layers
                if not traversed_tf_layers.issuperset(tf_in_layers):
                    continue

                new_tf_layers.add(tf_layer)
                logger.debug(f"Configuring {tf_layer.__class__.__name__} "
                             f"layer <{tf_layer.name}>")

                # === Add Layers ===
                if isinstance(tf_layer, tf.keras.layers.Add):
                    config = []

                    # concatenate incoming layer configs
                    for tf_in_layer in tf_in_layers:
                        config += configs_lookups[tf_in_layer]

                    # do not allow output Add layers
                    if len(tf_out_layers) == 0:
                        raise NotImplementedError(
                            'output Add layers not supported')

                    configs_lookups[tf_layer] = config


                # === Dense Layers ===
                elif isinstance(tf_layer, tf.keras.layers.Dense):

                    assert(len(tf_in_layers) == 1)
                    tf_in_layer = tf_in_layers[0]
                    in_configs = configs_lookups[tf_in_layer]

                    # configure layer
                    is_output = (len(tf_out_layers) == 0)
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=is_output,
                        has_activation=not tf_layer.activation is tf.keras.activations.linear,
                        neurons=self.create_neurons(tf_layer, pre_convert_output, 
                                                    is_output))

                    self.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Dense synapses
                            config.synapses.append(
                                InSynConfig(
                                    type=Dense,
                                    params={'weight': tf_layer.get_weights()[0]},
                                    source=in_config))

                        else:
                            for i in range(len(in_config.synapses)):
                                if in_config.synapses[i].type is AvgPool2D:
                                    # configure AvgPool2D -> Dense synapses
                                    config.synapses.append(
                                        InSynConfig(
                                            type=AvgPoolDense2D,
                                            params=in_config.synapses[i].params.copy(),
                                            source=in_config.synapses[i].source))
                                    config.synapses[-1].params.update({
                                        'weight': tf_layer.get_weights()[0]})

                                else:
                                    # fail if incoming (weighted) layer does not have activation
                                    if not in_config.has_activation:
                                        raise NotImplementedError(
                                            'weighted layers without activation not supported')

                    if config.has_activation or config.is_output:
                        config_steps.append(config)

                    configs_lookups[tf_layer] = [config]


                # === Conv2D Layers ===
                elif isinstance(tf_layer, tf.keras.layers.Conv2D):

                    assert(len(tf_in_layers) == 1)
                    tf_in_layer = tf_in_layers[0]
                    in_configs = configs_lookups[tf_in_layer]

                    # configure layer
                    is_output = (len(tf_out_layers) == 0)
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=is_output,
                        has_activation=not tf_layer.activation is tf.keras.activations.linear,
                        neurons=self.create_neurons(tf_layer, pre_convert_output,
                                                         is_output))

                    self.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Conv2D synapses
                            config.synapses.append(
                                InSynConfig(
                                    type=Conv2D,
                                    params={
                                        'filters': tf_layer.filters,
                                        'conv_size': tf_layer.kernel_size,
                                        'conv_strides': tf_layer.strides,
                                        'conv_padding': tf_layer.padding,
                                        'weight': tf_layer.get_weights()[0]},
                                    source=in_config))

                        else:
                            for i in range(len(in_config.synapses)):
                                if in_config.synapses[i].type is AvgPool2D:
                                    # configure AvgPool2D -> Conv2D synapses
                                    config.synapses.append(
                                        InSynConfig(
                                            type=AvgPoolConv2D,
                                            params=in_config.synapses[i].params.copy(),
                                            source=in_config.synapses[i].source))
                                    config.synapses[-1].params.update({
                                        'filters': tf_layer.filters,
                                        'conv_size': tf_layer.kernel_size,
                                        'conv_strides': tf_layer.strides,
                                        'conv_padding': tf_layer.padding,
                                        'weight': tf_layer.get_weights()[0]})

                                else:
                                    # fail if incoming (weighted) layer does not have activation
                                    if not in_config.has_activation:
                                        raise NotImplementedError(
                                            'weighted layers without activation not supported')

                    if config.has_activation or config.is_output:
                        config_steps.append(config)

                    configs_lookups[tf_layer] = [config]


                # === [Global]AveragePooling2D Layers ===
                elif isinstance(tf_layer, 
                                (tf.keras.layers.AveragePooling2D,
                                 tf.keras.layers.GlobalAveragePooling2D)):

                    assert(len(tf_in_layers) == 1)
                    tf_in_layer = tf_in_layers[0]
                    in_configs = configs_lookups[tf_in_layer]

                    # configure layer
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=len(tf_out_layers) == 0)

                    self.validate_tf_layer(tf_layer, config)

                    # do not allow output pooling layers
                    if config.is_output:
                        raise NotImplementedError(
                            'output pooling layers not supported')

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure AvgPool2D synapses
                            if isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
                                config.synapses.append(
                                    InSynConfig(
                                        type=AvgPool2D,
                                        params={'pool_size': tf_layer.pool_size,
                                                'pool_strides': tf_layer.strides},
                                        source=in_config))
                            elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
                                config.synapses.append(
                                    InSynConfig(
                                        type=AvgPool2D,
                                        params={'pool_size': tf_layer.input_shape[1:3],
                                                'pool_strides': None},
                                        source=in_config))

                        else:
                            # fail if incoming (weighted) layer does not have activation
                            if not in_config.has_activation:
                                raise NotImplementedError(
                                    'weighted layers without activation not supported')

                    configs_lookups[tf_layer] = [config]


                # === Activation Layers ===
                elif isinstance(tf_layer, tf_activation_layers):

                    assert(len(tf_in_layers) == 1)
                    tf_in_layer = tf_in_layers[0]
                    in_configs = configs_lookups[tf_in_layer]

                    # configure layer
                    is_output = (len(tf_out_layers) == 0)
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=is_output,
                        has_activation=True,
                        neurons=self.create_neurons(tf_layer, pre_convert_output,
                                                         is_output))

                    self.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:
                        if in_config.has_activation:
                            # configure Identity synapses
                            config.synapses.append(
                                InSynConfig(
                                    type=OneToOne,
                                    params={'weight': 1.0},
                                    source=in_config))

                        else:
                            for i in range(len(in_config.synapses)):
                                # copy incoming synapses
                                config.synapses.append(
                                    InSynConfig(
                                        type=in_config.synapses[i].type,
                                        params=in_config.synapses[i].params,
                                        source=in_config.synapses[i].source))

                    config_steps.append(config)

                    configs_lookups[tf_layer] = [config]


                # === Ignored Layers ===
                elif isinstance(tf_layer, tf_ignored_layers):

                    assert(len(tf_in_layers) == 1)
                    tf_in_layer = tf_in_layers[0]
                    in_configs = configs_lookups[tf_in_layer]

                    configs_lookups[tf_layer] = in_configs


                # === Unsupported Layers ===
                else:
                    raise NotImplementedError('{} layers not supported'.format(
                        tf_layer.__class__.__name__))


        # execute model build process
        mlg_pop_lookup = {}
        mlg_network_inputs = []
        mlg_model_outputs = []

        mlg_network = Network()
        with mlg_network:
            # for each build step
            for config in config_steps:
                # build population
                mlg_pop = Population(config.neurons, config.shape)
                
                if config.is_input:
                    mlg_network_inputs.append(mlg_pop)
                else:
                    # build connections
                    for s in config.synapses:
                        source = mlg_pop_lookup[s.source]
                        connectivity = s.type(**s.params)
                        
                        Connection(source, mlg_pop, connectivity)

                    if config.is_output:
                        mlg_model_outputs.append(mlg_pop)

                mlg_pop_lookup[config] = mlg_pop
        
        # Perform any pre-conversion tasks
        self.post_convert(mlg_network, mlg_network_inputs, mlg_model_outputs)
            
        return mlg_network, mlg_network_inputs, mlg_model_outputs
    
    def pre_convert(self, tf_model):
        pass
    
    def post_convert(self, mlg_network, mlg_network_inputs, mlg_model_outputs):
        pass
    
    def create_compiler(self, **kwargs):
        return InferenceCompiler(**kwargs)

    def validate_tf_layer(self, tf_layer, config):
        if isinstance(tf_layer, (tf.keras.layers.Dense,
                                 tf.keras.layers.Conv2D)):
            if tf_layer.use_bias:
                # no bias tensors allowed
                raise NotImplementedError("Bias tensors not supported")

            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError("Output layer must have ReLU "
                                              "or softmax activation")

            elif config.has_activation:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError("Hidden layers must have "
                                              "ReLU activation")

        elif isinstance(tf_layer, tf.keras.layers.Activation):
            if config.is_output:
                # ReLU and softmax activation allowd in output layers
                if (not tf_layer.activation is tf.keras.activations.relu and
                    not tf_layer.activation is tf.keras.activations.softmax):
                    raise NotImplementedError("Output layer must have "
                                              "ReLU or softmax activation")

            else:
                # ReLU activation allowed everywhere
                if not tf_layer.activation is tf.keras.activations.relu:
                    raise NotImplementedError("Hidden layers must have "
                                              "ReLU activation")

        elif isinstance(tf_layer, tf.keras.layers.ReLU):
            # ReLU activation allowed everywhere
            pass

        elif isinstance(tf_layer, tf.keras.layers.Softmax):
            # softmax activation only allowed for output layers
            if not config.is_output:
                raise NotImplementedError("Only output layers "
                                          "may use softmax")

        elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
            # global average pooling allowed
            pass
        elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
            if tf_layer.padding != 'valid':
                raise NotImplementedError("Only valid padding is supported "
                                          "for pooling layers")
        else:
            # no other layers allowed
            raise NotImplementedError(f"{tf_layer.__class__.__name__} "
                                      f"layers are not supported")
