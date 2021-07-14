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
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pygenn.genn_model import GeNNModel

from ml_genn.converters import Simple
from ml_genn.layers import InputLayer
from ml_genn.layers import Layer
from ml_genn.layers import Dense
from ml_genn.layers import AvePool2DDense
from ml_genn.layers import Conv2D
from ml_genn.layers import AvePool2DConv2D


class Model(object):
    """ML GeNN model class

    This class enables the creation of deep learning SNN models, and
    provides an interface for manipulating the underlying GeNN models.
    """

    def __init__(self, inputs, outputs, name='mlg_model'):
        """Initialise an ML GeNN model

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
        """Calculate depth of model's pipeline"""
        # **TODO** this only works for sequential models, branches need to be identified etc with e.g. ResNets
        return int(sum(hasattr(l.neurons, "pipelined")
                       for l in self.layers
                       if l not in self.outputs))

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

        supported_tf_layers = (
            tf.keras.layers.Dense,
            tf.keras.layers.Conv2D,
            tf.keras.layers.AveragePooling2D,
            tf.keras.layers.Flatten,
            tf.keras.layers.Dropout,
        )

        # Check model compatibility
        if not isinstance(tf_model, tf.keras.Sequential):
            raise NotImplementedError('{} models not supported'.format(
                tf_model.__class__.__name__))
        for tf_layer in tf_model.layers[:-1]:
            if not isinstance(tf_layer, supported_tf_layers):
                raise NotImplementedError('{} layers not supported'.format(
                    tf_layer.__class__.__name__))
            elif isinstance(tf_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                converter.validate_tf_layer(tf_layer)

        # Perform any pre-compilation tasks
        pre_compile_output = converter.pre_compile(tf_model)

        model_inputs = []
        model_outputs = []
        model_layers = []

        # === Input Layer ===
        layer = InputLayer('input', tf_model.input_shape[1:], 
                           converter.create_input_neurons(pre_compile_output))

        model_inputs.append(layer)
        model_layers.append(layer)
        previous_layer = layer
        pool_layer = None

        # For each TensorFlow model layer:
        for tf_layer in tf_model.layers:
            # === Flatten Layers ===
            if isinstance(tf_layer, tf.keras.layers.Flatten):
                print('ignoring Flatten layer <{}>'.format(tf_layer.name))

            # === Dropout Layers ===
            elif isinstance(tf_layer, tf.keras.layers.Dropout):
                print('ignoring Dropout layer <{}>'.format(tf_layer.name))

            # === Dense Layers ===
            elif isinstance(tf_layer, tf.keras.layers.Dense):
                if pool_layer is None:
                    print('converting Dense layer <{}>'.format(tf_layer.name))
                    layer = Dense(name=tf_layer.name, units=tf_layer.units,
                                  neurons=converter.create_neurons(tf_layer, pre_compile_output))
                else:
                    print('converting AveragePooling2D -> Dense layers <{}>'.format(tf_layer.name))
                    layer = AvePool2DDense(
                        name=tf_layer.name, units=tf_layer.units,
                        pool_size=pool_layer.pool_size,
                        pool_strides=pool_layer.strides,
                        pool_padding=pool_layer.padding,
                        connectivity_type=connectivity_type, 
                        neurons=converter.create_neurons(tf_layer, pre_compile_output))

                layer.connect([previous_layer])
                layer.set_weights(tf_layer.get_weights())

                model_layers.append(layer)
                previous_layer = layer
                pool_layer = None

            # === Conv2D Layers ===
            elif isinstance(tf_layer, tf.keras.layers.Conv2D):
                if pool_layer is None:
                    print('converting Conv2D layer <{}>'.format(tf_layer.name))
                    layer = Conv2D(
                        name=tf_layer.name, filters=tf_layer.filters,
                        conv_size=tf_layer.kernel_size,
                        conv_strides=tf_layer.strides,
                        conv_padding=tf_layer.padding,
                        connectivity_type=connectivity_type, 
                        neurons=converter.create_neurons(tf_layer, pre_compile_output))
                else:
                    print('converting AveragePooling2D -> Conv2D layers <{}>'.format(tf_layer.name))
                    layer = AvePool2DConv2D(
                        name=tf_layer.name, filters=tf_layer.filters,
                        pool_size=pool_layer.pool_size, conv_size=tf_layer.kernel_size,
                        pool_strides=pool_layer.strides, conv_strides=tf_layer.strides,
                        pool_padding=pool_layer.padding, conv_padding=tf_layer.padding,
                        connectivity_type=connectivity_type, 
                        neurons=converter.create_neurons(tf_layer, pre_compile_output))

                layer.connect([previous_layer])
                layer.set_weights(tf_layer.get_weights())

                model_layers.append(layer)
                previous_layer = layer
                pool_layer = None

            # === AveragePooling2D Layers ===
            elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
                print('deferring AveragePooling2D layer <{}>'.format(tf_layer.name))

                pool_layer = tf_layer

        model_outputs.append(previous_layer)

        model = Model(model_inputs, model_outputs, name=tf_model.name)

        # Compile model
        model.compile(**compile_kwargs)

        # Perform any post-compilation tasks
        converter.post_compile(model)

        return model











    @staticmethod
    def convert_tf_model_DAG(tf_model, converter=Simple(),
                             connectivity_type='procedural', **compile_kwargs):
        """Create a ML GeNN model from a TensorFlow model

        Args:
        tf_model  --  TensorFlow model to be converted

        Keyword args:
        input_type         --  type of input neurons (default: 'poisson')
        connectivity_type  --  type of synapses in GeNN (default: 'procedural')
        compile_kwargs     --  additional arguments to pass through to Model.compile
        """

        supported_tf_layers = (
            tf.keras.layers.Add,
            tf.keras.layers.InputLayer,
            tf.keras.layers.Dense,
            tf.keras.layers.Conv2D,
            tf.keras.layers.AveragePooling2D,
            tf.keras.layers.Flatten,
            tf.keras.layers.Dropout,
        )

        weighted_tf_layers = (
            tf.keras.layers.Dense,
            tf.keras.layers.Conv2D,
        )

        ignored_tf_layers = (
            tf.keras.layers.Flatten,
            tf.keras.layers.Dropout,
        )

        # Check model compatibility
        for tf_layer in tf_model.layers[:-1]:
            if not isinstance(tf_layer, supported_tf_layers):
                raise NotImplementedError('{} layers not supported'.format(
                    tf_layer.__class__.__name__))
            elif isinstance(tf_layer, weighted_tf_layers):
                converter.validate_tf_layer(tf_layer)

        # only traverse nodes belonging to this model
        tf_model_nodes = []
        for n in tf_model._nodes_by_depth.values():
            tf_model_nodes += n

        # get inbound and outbound layers
        tf_in_layers_all = {}
        tf_out_layers_all = {}
        for tf_layer in tf_model.layers:

            # find inbound layers
            tf_in_layers = []
            for in_layer in [node.inbound_layers for node in tf_layer.inbound_nodes
                             if node in tf_model_nodes]:
                if isinstance(in_layer, list):
                    tf_in_layers += in_layer
                else:
                    tf_in_layers.append(in_layer)
            tf_in_layers_all[tf_layer] = tf_in_layers

            # find outbound layers
            tf_out_layers = [node.outbound_layer for node in tf_layer.outbound_nodes
                             if node in tf_model_nodes]
            tf_out_layers_all[tf_layer] = tf_out_layers

        # configure model build process
        class InputLayerConfig(object):
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape
                self.is_output = False

        class LayerConfig(object):
            def __init__(self, name, syn_in, syn_w, syn_class, syn_args):
                self.name = name
                self.syn_in = syn_in
                self.syn_w = syn_w
                self.syn_class = syn_class
                self.syn_args = syn_args
                self.is_output = True

        config_lookup = {}
        config_steps = []

        new_tf_layers = set()
        traversed_tf_layers = set()

        # === Input Layers ===
        for input_name in tf_model.input_names:
            tf_layer = tf_model.get_layer(input_name)
            new_tf_layers.add(tf_layer)

            print('configuring Input layer <{}>'.format(tf_layer.name))

            name = tf_layer.name
            shape = tf_layer.input_shape[0][1:]

            step = InputLayerConfig(name, shape)
            config_lookup[tf_layer] = step
            config_steps.append(step)

        # while there are still layers to traverse
        while new_tf_layers:
            new_tf_layer = new_tf_layers.pop()
            traversed_tf_layers.add(new_tf_layer)

            # get next TF layer to configure
            for tf_layer in tf_out_layers_all[new_tf_layer]:
                tf_in_layers = tf_in_layers_all[tf_layer]
                tf_out_layers = tf_out_layers_all[tf_layer]

                # skip if we still need to configure inbound layers
                if not traversed_tf_layers.issuperset(tf_in_layers):
                    continue

                # add this layer to new layers list
                new_tf_layers.add(tf_layer)

                # === Dense Layers ===
                if isinstance(tf_layer, tf.keras.layers.Dense):
                    assert(len(tf_in_layers) == 1)
                    print('configuring Dense layer <{}>'.format(tf_layer.name))

                    name = tf_layer.name
                    syn_in = [config_lookup[tf_in_layers[0]]]
                    syn_w = [tf_layer.get_weights()]

                    if isinstance(tf_in_layers[0], tf.keras.layers.AveragePooling2D):
                        syn_class = [AvePool2DDense]
                        syn_args = [{
                            'name': tf_layer.name, 'units': tf_layer.units,
                            'pool_size': tf_in_layers[0].pool_size,
                            'pool_strides': tf_in_layers[0].strides,
                            'pool_padding': tf_in_layers[0].padding,
                            'connectivity_type': connectivity_type}]

                    else:
                        syn_class = [Dense]
                        syn_args = [{'name': tf_layer.name, 'units': tf_layer.units}]

                    config_lookup[tf_in_layers[0]].is_output = False

                    step = LayerConfig(name, syn_in, syn_w, syn_class, syn_args)
                    config_lookup[tf_layer] = step
                    config_steps.append(step)

                # === Conv2D Layers ===
                elif isinstance(tf_layer, tf.keras.layers.Conv2D):
                    assert(len(tf_in_layers) == 1)
                    print('configuring Conv2D layer <{}>'.format(tf_layer.name))

                    name = tf_layer.name
                    syn_in = [config_lookup[tf_in_layers[0]]]
                    syn_w = [tf_layer.get_weights()]

                    if isinstance(tf_in_layers[0], tf.keras.layers.AveragePooling2D):
                        syn_class = [AvePool2DConv2D]
                        syn_args = [{
                            'name': tf_layer.name, 'filters': tf_layer.filters,
                            'pool_size': tf_in_layers[0].pool_size, 'conv_size': tf_layer.kernel_size,
                            'pool_strides': tf_in_layers[0].strides, 'conv_strides': tf_layer.strides,
                            'pool_padding': tf_in_layers[0].padding, 'conv_padding': tf_layer.padding,
                            'connectivity_type': connectivity_type}]

                    else:
                        syn_class = [Conv2D]
                        syn_args = [{
                            'name': tf_layer.name, 'filters': tf_layer.filters,
                            'conv_size': tf_layer.kernel_size,
                            'conv_strides': tf_layer.strides,
                            'conv_padding': tf_layer.padding,
                            'connectivity_type': connectivity_type}]

                    config_lookup[tf_in_layers[0]].is_output = False

                    step = LayerConfig(name, syn_in, syn_w, syn_class, syn_args)
                    config_lookup[tf_layer] = step
                    config_steps.append(step)

                # === Add Layers ===
                elif isinstance(tf_layer, tf.keras.layers.Add):
                    assert(len(tf_in_layers) > 0)
                    print('configuring Add layer <{}>'.format(tf_layer.name))

                    name = tf_layer.name
                    syn_in = []
                    syn_w = []
                    syn_class = []
                    syn_args = []

                    for tf_in_layer in tf_in_layers:
                        # do not allow ignored layers before Add layer
                        if isinstance(tf_in_layer, ignored_tf_layers):
                            raise NotImplementedError(
                                '{} layers before Add layers not supported'.format(
                                    tf_in_layer.__class__.__name__))

                        syn_in.append(config_lookup[tf_in_layer].syn_in[0])
                        syn_w.append(config_lookup[tf_in_layer].syn_w[0])
                        syn_class.append(config_lookup[tf_in_layer].syn_class[0])
                        syn_args.append(config_lookup[tf_in_layer].syn_args[0])
                        config_steps.remove(config_lookup[tf_in_layer])

                        config_lookup[tf_in_layer].is_output = False

                    step = LayerConfig(name, syn_in, syn_w, syn_class, syn_args)
                    config_lookup[tf_layer] = step
                    config_steps.append(step)

                # === AveragePooling2D Layers ===
                elif isinstance(tf_layer, tf.keras.layers.AveragePooling2D):
                    assert(len(tf_in_layers) == 1)
                    print('configuring AveragePooling2D layer <{}>'.format(tf_layer.name))

                    # only allow weighted layers before AveragePooling2D layer
                    if not isinstance(tf_in_layers[0], weighted_tf_layers):
                        raise NotImplementedError(
                            '{} layers before AveragePooling2D layers not supported'.format(
                                tf_in_layers[0].__class__.__name__))

                    # do not allow pooling layers to be output layers
                    if len(tf_out_layers) == 0:
                        raise NotImplementedError(
                            'Pooling layers as output layers not supported'
                        )

                    config_lookup[tf_layer] = config_lookup[tf_in_layers[0]]

                # === Ignored Layers ===
                elif isinstance(tf_layer, ignored_tf_layers):
                    assert(len(tf_in_layers) == 1)
                    print('configuring {} layer <{}>'.format(
                        tf_layer.__class__.__name__, tf_layer.name))

                    # only allow weighted and Add layers before ignored layer
                    if not isinstance(tf_in_layers[0], weighted_tf_layers):
                        if not isinstance(tf_in_layers[0], tf.keras.layers.Add):
                            raise NotImplementedError(
                                '{} layers before {} layers not supported'.format(
                                    tf_in_layers[0].__class__.__name__,
                                    tf_layer.__class__.__name__))

                    config_lookup[tf_layer] = config_lookup[tf_in_layers[0]]


        # Perform any pre-compilation tasks
        pre_compile_output = converter.pre_compile(tf_model)

        # build the ML GeNN model
        mlg_layer_lookup = {}
        mlg_model_inputs = []
        mlg_model_outputs = []
        mlg_model_layers = []

        for step in config_steps:

            if isinstance(step, InputLayerConfig):
                mlg_layer = InputLayer(
                    name=step.name,
                    shape=step.shape,
                    neurons=converter.create_input_neurons(pre_compile_output))

                mlg_model_inputs.append(mlg_layer)

            elif isinstance(step, LayerConfig):
                mlg_layer = Layer(
                    name=step.name,
                    neurons=converter.create_neurons(tf_layer, pre_compile_output))

                sources = []
                synapses = []
                for syn_in, syn_class, syn_args in zip(step.syn_in, step.syn_class, step.syn_args):
                    sources.append(mlg_layer_lookup[syn_in])
                    synapses.append(syn_class(**syn_args))

                mlg_layer.connect(sources, synapses)
                mlg_layer.set_weights(step.syn_w)

                if step.is_output:
                    mlg_model_outputs.append(mlg_layer)

            mlg_layer_lookup[step] = mlg_layer
            mlg_model_layers.append(mlg_layer)

        # create ML GeNN model
        mlg_model = Model(mlg_model_inputs, mlg_model_outputs, name=tf_model.name)

        # Compile model
        mlg_model.compile(**compile_kwargs)

        # Perform any post-compilation tasks
        converter.post_compile(mlg_model)

        return mlg_model





    # TODO: do not allow multiple input or output layers when using Few Spike conversion

    # if isinstance(converter, FewSpike):
    #     raise NotImplementedError(
    #         'multiple input or output layers not supported for Few Spike conversion')
