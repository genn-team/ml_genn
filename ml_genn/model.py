"""ML GeNN model definition

This module provides the ``Model`` class to create deep learning SNN models,
and provides helper functions for operating the underlying GeNN model.

The ``Model`` class can also use a pre-trained TensorFlow model to function.
Such a model can be provided by calling the ``convert_tf_model`` method
with the TensorFlow model and optional parameters.

Example:
    The following is a minimal example which demonstrates the process of
    converting a TensorFlow model into a GeNN model and evaluating it:

        from ml_genn import Model

        ml_genn_model = Model.convert_tf_model(tensorflow_model)
        ml_genn_model.evaluate([test_data], [test_labels], 300.0)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pygenn.genn_model import GeNNModel

from ml_genn.converters import Simple

from ml_genn.layers import InputLayer
from ml_genn.layers import Layer

from ml_genn.layers import IdentitySynapses
from ml_genn.layers import DenseSynapses
from ml_genn.layers import Conv2DSynapses
from ml_genn.layers import AvePool2DSynapses
from ml_genn.layers import AvePool2DDenseSynapses
from ml_genn.layers import AvePool2DConv2DSynapses


class Model(object):
    """ML GeNN model class

    This class enables the creation of deep learning SNN models, and
    provides an interface for manipulating the underlying GeNN models.
    """

    def __init__(self, inputs, outputs, name='mlg_model'):
        """Initialise an ML GeNN model

        Args:
        inputs   --  list of network input layers
        outputs  --  list of network output layers

        Keyword args:
        name  --  name of the network (default: 'mlg_model')
        """

        self.set_network(inputs, outputs, name)


    def set_network(self, inputs, outputs, name='mlg_model'):
        """Construct an ML GeNN Model from a graph of Layers

        Args:
        inputs   --  list of network input layers
        outputs  --  list of network output layers

        Keyword args:
        name     --  name of the network (default: 'mlg_model')
        """

        self.name = name
        self.layers = []
        self.inputs = inputs
        self.outputs = outputs
        self.g_model = None

        # Construct topologically sorted list of layers
        new_layers = set(inputs)
        seen_synapses = set()
        while new_layers:
            layer = new_layers.pop()
            self.layers.append(layer)

            # Explore downstream layers whose upstream synapses have all been seen
            for downstream_synapse in layer.downstream_synapses:
                seen_synapses.add(downstream_synapse)
                if seen_synapses.issuperset(downstream_synapse.target().upstream_synapses):
                    new_layers.add(downstream_synapse.target())

        # Check that output layers are reachable from input layers
        if not all(output in self.layers for output in self.outputs):
            raise ValueError('output layers unreachable from input layers')


    def compile(self, dt=1.0, batch_size=1, rng_seed=0, reuse_genn_model=False,
                kernel_profiling=False, **genn_kwargs):
        """Compile this ML GeNN model into a GeNN model

        Keyword args:
        dt                --  model integration time step (default: 1.0)
        batch_size        --  number of models to run concurrently (default: 1)
        rng_seed          --  GeNN RNG seed (default: 0, meaning seed will be randomised at runtime)
        reuse_genn_model  --  Reuse existing compiled GeNN model (default: False)
        kernel_profiling  --  Build model with kernel profiling code (default: False)
        """

        # Define GeNN model
        self.g_model = GeNNModel('float', self.name, **genn_kwargs)
        self.g_model.dT = dt
        self.g_model.batch_size = batch_size
        self.g_model._model.set_seed(rng_seed)
        self.g_model.timing_enabled = kernel_profiling

        # Prepare each layer
        for layer in self.layers:
            layer.compile_neurons(self)
        for layer in self.layers:
            layer.compile_synapses(self)

        # Build and load GeNN model
        if os.name == 'nt':
            model_exists = os.path.isfile("./runner_Release.dll")
        else:
            model_exists = os.path.isfile('./' + self.name + '_CODE/librunner.so')
        if not reuse_genn_model or not model_exists:
            self.g_model.build()
        self.g_model.load()


    def set_input_batch(self, data_batch):
        """Set model input with a new batch of data

        Args:
        data_batch  --  list of data batches for each input layer
        """

        # Input sanity check
        if len(data_batch) != len(self.inputs):
            raise ValueError('data batch list length and input layer list length mismatch')

        for i in range(len(self.inputs)):
            self.inputs[i].set_input_batch(data_batch[i])


    def step_time(self, iterations=1):
        """Iterate the GeNN model a given number of steps

        Keyword args:
        iterations  --  number of iterations (default: 1)
        """

        for i in range(iterations):
            self.g_model.step_time()


    def reset(self):
        """Reset the GeNN model"""

        self.g_model.timestep = 0
        self.g_model.t = 0.0


    def evaluate(self, data, labels, time, save_samples=[]):
        """Evaluate the accuracy of a GeNN model

        Args:
        data          --  list of data for each input layer
        labels        --  list of labels for each output layer
        time          --  sample presentation time (msec)

        Keyword args:
        save_samples  --  list of sample indices to save spikes for (default: [])

        Returns:
        accuracy      --  percentage of correctly classified results
        spike_i       --  list of spike indices for each sample index in save_samples
        spike_t       --  list of spike times for each sample index in save_samples
        """

        # Input sanity check
        n_samples = data[0].shape[0]
        save_samples = list(set(save_samples))
        if len(data) != len(self.inputs):
            raise ValueError('data list length and input layer list length mismatch')
        if len(labels) != len(self.outputs):
            raise ValueError('label list length and output layer list length mismatch')
        if not all(x.shape[0] == n_samples for x in data + labels):
            raise ValueError('sample count mismatch in data and labels arrays')
        if any(i < 0 or i >= n_samples for i in save_samples):
            raise ValueError('one or more invalid save_samples value')

        n_correct = [0] * len(self.outputs)
        accuracy = [0] * len(self.outputs)
        all_spikes = [[[] for i,_ in enumerate(self.layers)] for s in save_samples]

        # Pad number of samples so pipeline can be flushed
        pipeline_depth = self.calc_pipeline_depth()
        padded_n_samples = n_samples + (pipeline_depth * self.g_model.batch_size)

        # Process batches
        progress = tqdm(total=n_samples)
        for batch_start in range(0, padded_n_samples, self.g_model.batch_size):
            # If any elements of this batch have data (rather than being entirely pipeline padding)
            if batch_start < n_samples:
                batch_end = min(batch_start + self.g_model.batch_size, n_samples)
                batch_data = [x[batch_start:batch_end] for x in data]

                save_samples_in_batch = [i for i in save_samples if batch_start <= i < batch_end]

                # Set new input
                self.set_input_batch(batch_data)

            # Reset timesteps etc
            self.reset()

            # Main simulation loop
            while self.g_model.t < time:
                # Step time
                self.step_time()

                # Save spikes
                for i in save_samples_in_batch:
                    k = save_samples.index(i)
                    batch_i = i - batch_start
                    for l, layer in enumerate(self.layers):
                        nrn = layer.neurons.nrn
                        nrn.pull_current_spikes_from_device()
                        all_spikes[k][l].append(np.copy(
                            nrn.current_spikes[batch_i] if self.g_model.batch_size > 1
                            else nrn.current_spikes))

            # If first input in batch has passed through
            if batch_start >= (pipeline_depth * self.g_model.batch_size):
                pipe_batch_start = batch_start - (pipeline_depth * self.g_model.batch_size)
                pipe_batch_end = min(pipe_batch_start + self.g_model.batch_size, n_samples)
                batch_labels = [y[pipe_batch_start:pipe_batch_end] for y in labels]

                # Compute accuracy
                for output_i in range(len(self.outputs)):
                    predictions = self.outputs[output_i].neurons.get_predictions(
                        pipe_batch_end - pipe_batch_start)
                    n_correct[output_i] += np.sum(predictions == batch_labels[output_i])
                    accuracy[output_i] = (n_correct[output_i] / pipe_batch_end) * 100

                progress.set_postfix_str('accuracy: {:2.2f}'.format(np.mean(accuracy)))
                progress.update(pipe_batch_end - pipe_batch_start)

        progress.close()

        # Create spike index and time lists
        spike_i = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        spike_t = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        for i in range(len(save_samples)):
            for j in range(len(self.layers)):
                spikes = all_spikes[i][j]
                spike_i[i][j] = np.concatenate(spikes)
                spike_t[i][j] = np.concatenate([np.ones_like(s) * i * self.g_model.dT for i, s in enumerate(spikes)])

        return accuracy, spike_i, spike_t

    def calc_pipeline_depth(self):
        # If none of the layers have the pipelined attribute, return 0
        if all(not hasattr(l.neurons, "pipelined")
               for l in self.layers if l not in self.outputs):
           return 0
       
        # If there are multiple inputs, give an error
        # **NOTE** inputs would have to be injected at different times to relax this
        if len(self.inputs) > 1:
            raise NotImplementedError("Pipelined models with multiple inputs "
                                      "are not currently supported")
        
        # If there are multiple outputs, give an error
        # **NOTE** outputs would need to be retrieved at different times to relax this
        if len(self.outputs) > 1:
            raise NotImplementedError("Pipelined models with multiple outputs "
                                      "are not currently supported")
                  
        # Check input is first layer (it should be)
        assert self.layers.index(self.inputs[0]) == 0
        
        # Initialise layer distances to maxsize for all layers aside from input
        shortest_distance = {layer: (0 if layer == self.inputs[0] else sys.maxsize)
                             for layer in self.layers}
       
        # Initialiser list of best layer predecessors
        best_predecessor = {layer: None for layer in self.layers}
        
        # Loop through layers
        for v in self.layers:
            # Loop through layer's outgoing edges
            for s in v.downstream_synapses:
                # Edges add one to pipeline depth if target is 
                # not an output and its neuron model isn't pipelined
                u = s.target()
                weight = 1 if u not in self.outputs and hasattr(u.neurons, "pipelined") else 0
                
                # If this results in a shorter route, return
                if shortest_distance[u] > shortest_distance[v]:
                    shortest_distance[u] = shortest_distance[v] + weight
                    best_predecessor[u] = v
        
        # Return shortest distance to output
        return shortest_distance[self.outputs[0]]

    def get_kernel_times(self):
        """Get total kernel run times"""

        return {
            'init_time': self.g_model.init_time,
            'init_sparse_time': self.g_model.init_sparse_time,
            'neuron_update_time': self.g_model.neuron_update_time,
            'presynaptic_update_time': self.g_model.presynaptic_update_time,
            'postsynaptic_update_time': self.g_model.postsynaptic_update_time,
            'synapse_dynamics_time': self.g_model.synapse_dynamics_time,
        }

    def summary(self):
        """Print a summary of this model"""

        # layers should already be topologically sorted
        print('===== Summary of {} ====='.format(self.name))

        for l in self.layers:
            print('\nname: {},  shape: {},  type: {},'.format(
                l.name, l.shape, l.__class__.__name__))

            if isinstance(l, Layer):
                print('incoming: {}'.format(
                    {s.source().name: s.__class__.__name__ for s in l.upstream_synapses}))

    @staticmethod
    def convert_tf_model(tf_model, converter=Simple(),
                         connectivity_type='procedural', **compile_kwargs):
        """Create a ML GeNN model from a TensorFlow model

        Args:
        tf_model  --  TensorFlow model to be converted

        Keyword args:
        input_type         --  type of input neurons (default: 'poisson')
        connectivity_type  --  type of synapses in GeNN (default: 'procedural')
        compile_kwargs     --  additional arguments to pass through to Model.compile
        """

        tf_activation_layers = (
            tf.keras.layers.ReLU)

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


        # Perform any pre-compilation tasks
        pre_compile_output = converter.pre_compile(tf_model)

        # configure model build process
        class LayerConfig(object):
            def __init__(self):
                self.is_input = False
                self.is_output = False
                self.has_activation = False
                self.name = None
                self.shape = None
                self.neurons = None
                self.synapses_class = []
                self.synapses_params = []
                self.synapses_source = []
                self.synapses_weights = []

        config_steps = []
        configs_lookups = {}
        new_tf_layers = set()
        traversed_tf_layers = set()

        # get and check input layers
        if isinstance(tf_model, tf.keras.models.Sequential):
            # In TF Sequential models, the InputLayer is not stored in the model object,
            # so we must traverse back through nodes to find the input layer's outputs.
            tf_in_layers = tf_in_layers_all[tf_model.layers[0]]
            assert(len(tf_in_layers) == 1)
            tf_out_layers = [n.outbound_layer for n in tf_in_layers[0].outbound_nodes
                             if n in tf_model_nodes]
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


        # === Input Layers ===
        for tf_layer in tf_in_layers:
            new_tf_layers.add(tf_layer)
            print('configuring Input layer <{}>'.format(tf_layer.name))

            # configure layer
            config = LayerConfig()
            config.is_input = True
            config.has_activation = True
            config.name = tf_layer.name
            config.shape = tf_layer.output_shape[0][1:]
            config.neurons = converter.create_input_neurons(pre_compile_output)

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
                print('configuring {} layer <{}>'.format(
                    tf_layer.__class__.__name__, tf_layer.name))


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
                    config = LayerConfig()
                    config.is_output = len(tf_out_layers) == 0
                    config.name = tf_layer.name
                    config.shape = tf_layer.output_shape[1:]
                    config.neurons = converter.create_neurons(tf_layer, pre_compile_output)
                    if not tf_layer.activation is tf.keras.activations.linear:
                        config.has_activation = True

                    converter.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Dense synapses
                            config.synapses_class.append(DenseSynapses)
                            config.synapses_params.append({'units': tf_layer.units})
                            config.synapses_source.append(in_config)
                            config.synapses_weights.append(tf_layer.get_weights()[0])

                        else:
                            for i in range(len(in_config.synapses_class)):

                                if in_config.synapses_class[i] is AvePool2DSynapses:
                                    # configure AvePool2D -> Dense synapses
                                    config.synapses_class.append(AvePool2DDenseSynapses)
                                    config.synapses_params.append(in_config.synapses_params[i].copy())
                                    config.synapses_params[-1].update({'units': tf_layer.units})
                                    config.synapses_source.append(in_config.synapses_source[i])
                                    config.synapses_weights.append(tf_layer.get_weights()[0])

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
                    config = LayerConfig()
                    config.is_output = len(tf_out_layers) == 0
                    config.name = tf_layer.name
                    config.shape = tf_layer.output_shape[1:]
                    config.neurons = converter.create_neurons(tf_layer, pre_compile_output)
                    if not tf_layer.activation is tf.keras.activations.linear:
                        config.has_activation = True

                    converter.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Conv2D synapses
                            config.synapses_class.append(Conv2DSynapses)
                            config.synapses_params.append({
                                'filters': tf_layer.filters,
                                'conv_size': tf_layer.kernel_size,
                                'conv_strides': tf_layer.strides,
                                'conv_padding': tf_layer.padding,
                                'connectivity_type': connectivity_type})
                            config.synapses_source.append(in_config)
                            config.synapses_weights.append(tf_layer.get_weights()[0])

                        else:
                            for i in range(len(in_config.synapses_class)):

                                if in_config.synapses_class[i] is AvePool2DSynapses:
                                    # configure AvePool2D -> Conv2D synapses
                                    config.synapses_class.append(AvePool2DConv2DSynapses)
                                    config.synapses_params.append(in_config.synapses_params[i].copy())
                                    config.synapses_params[-1].update({
                                        'filters': tf_layer.filters,
                                        'conv_size': tf_layer.kernel_size,
                                        'conv_strides': tf_layer.strides,
                                        'conv_padding': tf_layer.padding,
                                        'connectivity_type': connectivity_type})
                                    config.synapses_source.append(in_config.synapses_source[i])
                                    config.synapses_weights.append(tf_layer.get_weights()[0])

                                else:
                                    # fail if incoming (weighted) layer does not have activation
                                    if not in_config.has_activation:
                                        raise NotImplementedError(
                                            'weighted layers without activation not supported')

                    if config.has_activation or config.is_output:
                        config_steps.append(config)

                    configs_lookups[tf_layer] = [config]


                # === [Global]AveragePooling2D Layers ===
                elif isinstance(tf_layer, (
                        tf.keras.layers.AveragePooling2D,
                        tf.keras.layers.GlobalAveragePooling2D)):

                    assert(len(tf_in_layers) == 1)
                    tf_in_layer = tf_in_layers[0]
                    in_configs = configs_lookups[tf_in_layer]

                    # configure layer
                    config = LayerConfig()
                    config.is_output = len(tf_out_layers) == 0
                    config.name = tf_layer.name
                    config.shape = tf_layer.output_shape[1:]

                    converter.validate_tf_layer(tf_layer, config)

                    # do not allow output pooling layers
                    if config.is_output:
                        raise NotImplementedError(
                            'output pooling layers not supported')

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure AvePool2D synapses
                            config.synapses_class.append(AvePool2DSynapses)
                            if isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
                                config.synapses_params.append({
                                    'pool_size': tf_layer.pool_size,
                                    'pool_strides': tf_layer.strides,
                                    'pool_padding': tf_layer.padding})
                            elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
                                config.synapses_params.append({
                                    'pool_size': tf_layer.input_shape[1:3],
                                    'pool_strides': None,
                                    'pool_padding': 'valid'})
                            config.synapses_source.append(in_config)
                            config.synapses_weights.append(None)

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
                    config = LayerConfig()
                    config.is_output = len(tf_out_layers) == 0
                    config.name = tf_layer.name
                    config.shape = tf_layer.output_shape[1:]
                    config.has_activation = True
                    config.neurons = converter.create_neurons(tf_layer, pre_compile_output)

                    converter.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Identity synapses
                            config.synapses_class.append(IdentitySynapses)
                            config.synapses_params.append({})
                            config.synapses_source.append(in_config)
                            config.synapses_weights.append(None)

                        else:
                            for i in range(len(in_config.synapses_class)):
                                # copy incoming synapses
                                config.synapses_class.append(in_config.synapses_class[i])
                                config.synapses_params.append(in_config.synapses_params[i])
                                config.synapses_source.append(in_config.synapses_source[i])
                                config.synapses_weights.append(in_config.synapses_weights[i])

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
        mlg_layer_lookup = {}
        mlg_model_inputs = []
        mlg_model_outputs = []

        # for each build step
        for config in config_steps:

            if config.is_input:
                # build layer
                mlg_layer = InputLayer(config.name, config.shape, config.neurons)

                mlg_model_inputs.append(mlg_layer)

            else:
                # build layer
                mlg_layer = Layer(config.name, config.neurons)

                # build synapses
                sources = []
                synapses = []
                weights = config.synapses_weights
                for i in range(len(config.synapses_class)):
                    sources.append(mlg_layer_lookup[config.synapses_source[i]])
                    synapses.append(config.synapses_class[i](**config.synapses_params[i]))
                mlg_layer.connect(sources, synapses)
                mlg_layer.set_weights(weights)

                if config.is_output:
                    mlg_model_outputs.append(mlg_layer)

            mlg_layer_lookup[config] = mlg_layer


        # create model
        mlg_model = Model(mlg_model_inputs, mlg_model_outputs, name=tf_model.name)

        # Compile model
        mlg_model.compile(**compile_kwargs)

        # Perform any post-compilation tasks
        converter.post_compile(mlg_model)

        return mlg_model
