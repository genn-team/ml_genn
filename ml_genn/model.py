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
from collections import namedtuple, deque

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

        # Construct topologically sorted list of layers (Kahn's algorithm as described here: https://en.wikipedia.org/wiki/Topological_sorting)
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


    def evaluate_batched(self, data, time, save_samples=[]):
        """Evaluate the accuracy of a GeNN model

        Args:
        data          --  an [[x_1, ..., x_m], [y_1, ..., y_n]] batch dataset
        time          --  sample presentation time (msec)

        Keyword args:
        save_samples  --  list of sample indices to save spikes for (default: [])

        Returns:
        accuracy      --  percentage of correctly classified results
        spike_i       --  list of spike indices for each sample index in save_samples
        spike_t       --  list of spike times for each sample index in save_samples
        """

        n_complete = [0] * len(self.outputs)
        n_correct = [0] * len(self.outputs)
        accuracy = [0] * len(self.outputs)

        save_samples = list(set(save_samples))
        all_spikes = [[[] for i,_ in enumerate(self.layers)] for s in save_samples]

        # Pipeline queues (for each output layer)
        pipeline_depth = [l.pipeline_depth if hasattr(l, 'pipeline_depth') else 0 for l in self.outputs]
        pipeline_y_queue = [deque(maxlen=depth + 1) for depth in pipeline_depth]

        # Get batch iterator
        data_remaining = True
        data_iter = iter(data)
        batch_x, batch_y = next(data_iter)
        if not isinstance(batch_x, tuple):
            batch_x = (batch_x, )
        if not isinstance(batch_y, tuple):
            batch_y = (batch_y, )
        batch_i = 0

        # Check number of x and y elements match number of inputs and outputs
        if len(batch_x) != len(self.inputs):
            raise ValueError('input layer and x count mismatch')
        if len(batch_y) != len(self.outputs):
            raise ValueError('output layer and y count mismatch')

        # Process batches
        progress = tqdm()
        while data_remaining or any(len(q) > 0 for q in pipeline_y_queue):

            if data_remaining:
                batch_x = [np.asarray(d) for d in batch_x]
                batch_size = batch_x[0].shape[0]
                batch_start = batch_i * self.g_model.batch_size
                batch_end = batch_start + batch_size
                save_samples_in_batch = [i for i in save_samples if batch_start <= i < batch_end]

                # Queue labels for pipelining
                batch_y = [np.asarray(l) for l in batch_y]
                for output_i in range(len(self.outputs)):
                    pipeline_y_queue[output_i].append(batch_y[output_i])

                # Set new input
                self.set_input_batch(batch_x)

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

            for output_i in range(len(self.outputs)):
                # If input has passed through pipeline to this output
                if batch_i >= pipeline_depth[output_i] and len(pipeline_y_queue[output_i]) > 0:
                    pipe_batch_i = batch_i - pipeline_depth[output_i]
                    pipe_batch_y = pipeline_y_queue[output_i].popleft()
                    pipe_batch_size = pipe_batch_y.shape[0]
                    pipe_batch_start = pipe_batch_i * self.g_model.batch_size
                    pipe_batch_end = pipe_batch_start + pipe_batch_size

                    # Compute accuracy
                    predictions = self.outputs[output_i].neurons.get_predictions(pipe_batch_size)
                    if pipe_batch_y.shape != predictions.shape:
                        pipe_batch_y = [np.argmax(i) for i in pipe_batch_y]

                    n_complete[output_i] += pipe_batch_size
                    n_correct[output_i] += np.sum(predictions == pipe_batch_y)
                    accuracy[output_i] = (n_correct[output_i] / pipe_batch_end) * 100

            progress.set_postfix_str('accuracy: {:2.2f}'.format(np.mean(accuracy)))
            if data_remaining:
                progress.update(batch_size)

            try:
                batch_x, batch_y = next(data_iter)
                if not isinstance(batch_x, tuple):
                    batch_x = (batch_x, )
                if not isinstance(batch_y, tuple):
                    batch_y = (batch_y, )
            except StopIteration:
                data_remaining = False
            batch_i += 1

        progress.close()
        assert all(len(q) == 0 for q in pipeline_y_queue)

        # Create spike index and time lists
        spike_i = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        spike_t = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        for i in range(len(save_samples)):
            for j in range(len(self.layers)):
                spikes = all_spikes[i][j]
                spike_i[i][j] = np.concatenate(spikes)
                spike_t[i][j] = np.concatenate([np.ones_like(s) * i * self.g_model.dT for i, s in enumerate(spikes)])

        return accuracy, spike_i, spike_t


    def evaluate(self, x, y, time, save_samples=[]):
        """Evaluate the accuracy of a GeNN model

        Args:
        x             --  tuple of data for each input layer
        y             --  tuple of labels for each output layer
        time          --  sample presentation time (msec)

        Keyword args:
        save_samples  --  list of sample indices to save spikes for (default: [])

        Returns:
        accuracy      --  percentage of correctly classified results
        spike_i       --  list of spike indices for each sample index in save_samples
        spike_t       --  list of spike times for each sample index in save_samples
        """

        # Wrap singular x or y in tuples
        if not isinstance(x, tuple):
            x = (x, )
        if not isinstance(y, tuple):
            y = (y, )

        # Batch all x and y
        batch_size = self.g_model.batch_size
        x = tuple(np.split(xx, np.arange(batch_size, len(xx), batch_size)) for xx in x)
        y = tuple(np.split(yy, np.arange(batch_size, len(yy), batch_size)) for yy in y)
        data = zip(zip(*x), zip(*y))

        # Pass to evaluate_batched
        accuracy, spike_i, spike_t = self.evaluate_batched(data, time, save_samples)

        return accuracy, spike_i, spike_t


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
        pre_convert_output = converter.pre_convert(tf_model)

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

        InSynConfig = namedtuple('InSynconfig', ['type', 'params', 'source', 'weights'])

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


        # === Input Layers ===
        for tf_layer in tf_in_layers:
            new_tf_layers.add(tf_layer)
            print('configuring Input layer <{}>'.format(tf_layer.name))

            # configure layer
            config = LayerConfig(
                tf_layer.name, tf_layer.output_shape[0][1:],
                is_input=True, has_activation=True,
                neurons=converter.create_input_neurons(pre_convert_output))

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
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=len(tf_out_layers) == 0,
                        has_activation=not tf_layer.activation is tf.keras.activations.linear,
                        neurons=converter.create_neurons(tf_layer, pre_convert_output))

                    converter.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Dense synapses
                            config.synapses.append(InSynConfig(
                                type=DenseSynapses,
                                params={'units': tf_layer.units},
                                source=in_config,
                                weights=tf_layer.get_weights()[0]))

                        else:
                            for i in range(len(in_config.synapses)):

                                if in_config.synapses[i].type is AvePool2DSynapses:
                                    # configure AvePool2D -> Dense synapses
                                    config.synapses.append(InSynConfig(
                                        type=AvePool2DDenseSynapses,
                                        params=in_config.synapses[i].params.copy(),
                                        source=in_config.synapses[i].source,
                                        weights=tf_layer.get_weights()[0]))
                                    config.synapses[-1].params.update({
                                        'units': tf_layer.units})

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
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=len(tf_out_layers) == 0,
                        has_activation=not tf_layer.activation is tf.keras.activations.linear,
                        neurons=converter.create_neurons(tf_layer, pre_convert_output))

                    converter.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Conv2D synapses
                            config.synapses.append(InSynConfig(
                                type=Conv2DSynapses,
                                params={
                                    'filters': tf_layer.filters,
                                    'conv_size': tf_layer.kernel_size,
                                    'conv_strides': tf_layer.strides,
                                    'conv_padding': tf_layer.padding,
                                    'connectivity_type': connectivity_type},
                                source=in_config,
                                weights=tf_layer.get_weights()[0]))

                        else:
                            for i in range(len(in_config.synapses)):

                                if in_config.synapses[i].type is AvePool2DSynapses:
                                    # configure AvePool2D -> Conv2D synapses
                                    config.synapses.append(InSynConfig(
                                        type=AvePool2DConv2DSynapses,
                                        params=in_config.synapses[i].params.copy(),
                                        source=in_config.synapses[i].source,
                                        weights=tf_layer.get_weights()[0]))
                                    config.synapses[-1].params.update({
                                        'filters': tf_layer.filters,
                                        'conv_size': tf_layer.kernel_size,
                                        'conv_strides': tf_layer.strides,
                                        'conv_padding': tf_layer.padding,
                                        'connectivity_type': connectivity_type})

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
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=len(tf_out_layers) == 0)

                    converter.validate_tf_layer(tf_layer, config)

                    # do not allow output pooling layers
                    if config.is_output:
                        raise NotImplementedError(
                            'output pooling layers not supported')

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure AvePool2D synapses
                            if isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
                                config.synapses.append(InSynConfig(
                                    type=AvePool2DSynapses,
                                    params={
                                        'pool_size': tf_layer.pool_size,
                                        'pool_strides': tf_layer.strides,
                                        'connectivity_type': connectivity_type},
                                    source=in_config,
                                    weights=None))
                            elif isinstance(tf_layer, tf.keras.layers.GlobalAveragePooling2D):
                                config.synapses.append(InSynConfig(
                                    type=AvePool2DSynapses,
                                    params={
                                        'pool_size': tf_layer.input_shape[1:3],
                                        'pool_strides': None,
                                        'connectivity_type': connectivity_type},
                                    source=in_config,
                                    weights=None))

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
                    config = LayerConfig(
                        tf_layer.name, tf_layer.output_shape[1:],
                        is_output=len(tf_out_layers) == 0,
                        has_activation=True,
                        neurons=converter.create_neurons(tf_layer, pre_convert_output))

                    converter.validate_tf_layer(tf_layer, config)

                    # configure synapses
                    for in_config in in_configs:

                        if in_config.has_activation:
                            # configure Identity synapses
                            config.synapses.append(InSynConfig(
                                type=IdentitySynapses,
                                params={'connectivity_type': connectivity_type},
                                source=in_config,
                                weights=None))

                        else:
                            for i in range(len(in_config.synapses)):
                                # copy incoming synapses
                                config.synapses.append(InSynConfig(
                                    type=in_config.synapses[i].type,
                                    params=in_config.synapses[i].params,
                                    source=in_config.synapses[i].source,
                                    weights=in_config.synapses[i].weights))

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
                sources = [mlg_layer_lookup[s.source] for s in config.synapses]
                synapses = [s.type(**s.params) for s in config.synapses]
                mlg_layer.connect(sources, synapses)
                weights = [s.weights for s in config.synapses]
                mlg_layer.set_weights(weights)

                if config.is_output:
                    mlg_model_outputs.append(mlg_layer)

            mlg_layer_lookup[config] = mlg_layer


        # create model
        mlg_model = Model(mlg_model_inputs, mlg_model_outputs, name=tf_model.name)
    
        # Perform any pre-compilation tasks
        converter.pre_compile(mlg_model)
        
        # Compile model
        mlg_model.compile(**compile_kwargs)

        # Perform any post-compilation tasks
        converter.post_compile(mlg_model)

        return mlg_model
