"""TensorGeNN model definition

This module provides TGModel class to convert TensorFlow models into GeNN
models, and provides helper functions for operating the resulting GeNN model.

A ``TGModel`` object can use a pre-trained TensorFlow model to function.
Such a model can be provided by calling the ``convert_tf_model`` method
with the TensorFlow model and optional parameters.

Example:
    The following is a minimal example which demonstrates the process of
    converting a TensorFlow model into a GeNN model and evaluating it::

        from tensor_genn import TGModel

        tensorgenn_model = TGmodel()
        tensorgenn_model.convert_tf_model(tensorflow_model)
        tensorgenn_model.compile()
        tensorgenn_model.evaluate(test_data, test_labels)
"""

import numpy as np
import tensorflow as tf
from math import ceil
from enum import Enum
from tqdm import tqdm
from pygenn import genn_model, genn_wrapper

from tensor_genn.genn_models import if_model, if_init
from tensor_genn.genn_models import if_input_model, if_input_init
from tensor_genn.genn_models import poisson_input_model, poisson_input_init
from tensor_genn.genn_models import spike_input_model, spike_input_init


class InputType(Enum):
    IF = 'if'
    POISSON = 'poisson'
    SPIKE = 'spike'

    def __str__(self):
        return self.value


class TGModel(object):
    """TensorGeNN model class

    This class converts fully trained TensorFlow models into GeNN models,
    and provides an interface for manipulating converted models.
    """

    def __init__(self, name='tg_model'):
        """Initialise a TensorGeNN model"""

        self.batch_size = None
        self.tf_model = None
        self.name = name
        self.g_model = None
        self.layer_names = None
        self.weight_vals = None
        self.weight_conn = None
        self.thresholds = None


    def convert_tf_model(self, tf_model):
        """Convert from a TensorFlow model

        Args:
        tf_model  --  TensorFlow model to be converted
        """

        supported_layers = (
            tf.keras.layers.Dense,
            tf.keras.layers.Conv2D,
            tf.keras.layers.AveragePooling2D,
            tf.keras.layers.Flatten,
            tf.keras.layers.Dropout,
        )

        # Check model compatibility
        if not isinstance(tf_model, tf.keras.Sequential):
            raise NotImplementedError('{} models not supported'.format(type(tf_model)))
        for layer in tf_model.layers[:-1]:
            if not isinstance(layer, supported_layers):
                raise NotImplementedError('{} layers not supported'.format(type(layer)))
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.activation != tf.keras.activations.relu:
                    raise NotImplementedError('{} activation not supported'.format(type(layer.activation)))
                if layer.use_bias == True:
                    raise NotImplementedError('bias tensors not supported')

        self.tf_model = tf_model
        self.name = tf_model.name
        self.layer_names = []
        self.weight_vals = []
        self.weight_conn = []
        self.thresholds = []
        deferred_vals = None
        deferred_conn = None

        # For each TensorFlow model layer:
        for layer in tf_model.layers:

            # === Flatten Layers ===
            if isinstance(layer, tf.keras.layers.Flatten):
                print('ignoring layer <{}>'.format(layer.name))
                continue

            # === Dropout Layers ===
            if isinstance(layer, tf.keras.layers.Dropout):
                print('ignoring layer <{}>'.format(layer.name))
                continue

            # === Dense Layers ===
            elif isinstance(layer, tf.keras.layers.Dense):
                print('creating GeNN weights for layer <{}>'.format(layer.name))
                tf_vals = layer.get_weights()[0]
                g_vals = tf_vals.copy()
                g_conn = np.ones(g_vals.shape, dtype=np.bool)
                defer_weights = False

            # === Conv2D Layers ===
            elif isinstance(layer, tf.keras.layers.Conv2D):
                print('creating GeNN weights for layer <{}>'.format(layer.name))
                tf_vals = layer.get_weights()[0]
                g_vals = np.zeros((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])))
                g_conn = np.zeros(g_vals.shape, dtype=np.bool)
                defer_weights = False

                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # Stride start index iterators.
                if layer.padding == 'valid':
                    stride_rows = range(0, ih - kh + 1, sh)
                    stride_cols = range(0, iw - kw + 1, sw)
                elif layer.padding == 'same':
                    stride_rows = range(0 - kh // 2, ih - kh // 2, sh)
                    stride_cols = range(0 - kw // 2, iw - kw // 2, sw)

                # For each kernel (all-to-all input -> output channels):
                for in_channel in range(ic):
                    for out_channel in range(oc):
                        tf_channel_vals = tf_vals[:, :, in_channel, out_channel]
                        g_channel_vals = g_vals[in_channel::ic, out_channel::oc]
                        g_channel_conn = g_conn[in_channel::ic, out_channel::oc]

                        # For each output neuron:
                        for out_row, stride_row in enumerate(stride_rows):
                            for out_col, stride_col in enumerate(stride_cols):
                                g_out_vals = g_channel_vals[:, out_row * ow + out_col]
                                g_out_vals.shape = (ih, iw)
                                g_out_conn = g_channel_conn[:, out_row * ow + out_col]
                                g_out_conn.shape = (ih, iw)

                                # Get a kernel stride view in tf_channel_vals.
                                tf_kern_T = 0 - min(stride_row, 0)
                                tf_kern_B = kh - max(stride_row + kh - ih, 0)
                                tf_kern_L = 0 - min(stride_col, 0)
                                tf_kern_R = kw - max(stride_col + kw - iw, 0)
                                tf_stride_vals = tf_channel_vals[tf_kern_T:tf_kern_B, tf_kern_L:tf_kern_R]

                                # Get a kernel stride view in g_out_vals.
                                g_kern_T = max(stride_row, 0)
                                g_kern_B = min(stride_row + kh, ih)
                                g_kern_L = max(stride_col, 0)
                                g_kern_R = min(stride_col + kw, iw)
                                g_stride_vals = g_out_vals[g_kern_T:g_kern_B, g_kern_L:g_kern_R]
                                g_stride_conn = g_out_conn[g_kern_T:g_kern_B, g_kern_L:g_kern_R]

                                # Set weights for this stride.
                                g_stride_vals[:] = tf_stride_vals
                                g_stride_conn[:] = True

            # === AveragePooling2D Layers ===
            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                print('defer weights for layer <{}>'.format(layer.name))
                g_vals = np.zeros((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])))
                g_conn = np.zeros(g_vals.shape, dtype=np.bool)
                defer_weights = True

                ph, pw = layer.pool_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # Stride start index iterators.
                if layer.padding == 'valid':
                    stride_rows = range(0, ih - ph + 1, sh)
                    stride_cols = range(0, iw - pw + 1, sw)
                elif layer.padding == 'same':
                    stride_rows = range(0 - ph // 2, ih - ph // 2, sh)
                    stride_cols = range(0 - pw // 2, iw - pw // 2, sw)

                # For each pool (one-to-one input -> output channels):
                for channel in range(ic):
                    g_pool_vals = g_vals[channel::ic, channel::ic]
                    g_pool_conn = g_conn[channel::ic, channel::ic]

                    # For each output neuron:
                    for out_row, stride_row in enumerate(stride_rows):
                        for out_col, stride_col in enumerate(stride_cols):
                            g_out_vals = g_pool_vals[:, out_row * ow + out_col]
                            g_out_vals.shape = (ih, iw)
                            g_out_conn = g_pool_conn[:, out_row * ow + out_col]
                            g_out_conn.shape = (ih, iw)

                            # Get a pool stride view in g_out_vals.
                            g_pool_T = max(stride_row, 0)
                            g_pool_B = min(stride_row + ph, ih)
                            g_pool_L = max(stride_col, 0)
                            g_pool_R = min(stride_col + pw, iw)
                            g_stride_vals = g_out_vals[g_pool_T:g_pool_B, g_pool_L:g_pool_R]
                            g_stride_conn = g_out_conn[g_pool_T:g_pool_B, g_pool_L:g_pool_R]

                            # Set weights for this stride.
                            g_stride_vals[:] = 1.0 / g_stride_vals.size
                            g_stride_conn[:] = True

            # === Combine Deferred Weights ===
            if deferred_vals is not None:
                print('combining deferred weights with GeNN weights for layer <{}>'.format(layer.name))
                new_vals = np.zeros((deferred_vals.shape[0], g_vals.shape[1]))
                new_conn = np.zeros(new_vals.shape, dtype=np.bool)

                # For each input channel:
                for in_channel in range(ic):
                    # Note: it is assumed that the deferred weight matrix maps input
                    # channel i one-to-one to output channel i, for all i in [0, ic).
                    new_in_channel_vals = new_vals[in_channel::ic, :]
                    new_in_channel_conn = new_conn[in_channel::ic, :]
                    deferred_in_channel_vals = deferred_vals[in_channel::ic, in_channel::ic]
                    deferred_in_channel_conn = deferred_conn[in_channel::ic, in_channel::ic]
                    g_in_channel_vals = g_vals[in_channel::ic, :]
                    g_in_channel_conn = g_conn[in_channel::ic, :]

                    # Set weights to dot product of deferred and new weights.
                    new_in_channel_vals[:] = np.dot(deferred_in_channel_vals, g_in_channel_vals)
                    new_in_channel_conn[:] = np.dot(deferred_in_channel_conn, g_in_channel_conn)

                # Update weights.
                g_vals = new_vals
                g_conn = new_conn

            # === Append Weights ===
            if defer_weights:
                # Defer weights to next layer.
                deferred_vals = g_vals
                deferred_conn = g_conn
            else:
                # Append weights for this layer.
                self.layer_names.append(layer.name)
                self.weight_vals.append(g_vals)
                self.weight_conn.append(g_conn)
                self.thresholds.append(1.0)
                deferred_vals = None
                deferred_conn = None


    def compile(self, batch_size=1, dt=1.0, input_type=InputType.IF, rate_factor=1.0, rng_seed=0):
        """Compile this TensorGeNN model into a GeNN model

        Keyword args:
        batch_size   --  number of models to run concurrently (default: 1)
        dt           --  model integration time step (default: 1.0)
        input_type   --  type of input neurons (default: 'if')
        rate_factor  --  scale firing rate if input_type is 'poisson' (default: 1.0)
        rng_seed     --  GeNN RNG seed (default: 0, meaning choose a random seed)
        """

        # Define GeNN model
        self.batch_size = batch_size
        g_model = genn_model.GeNNModel('float', self.name)
        g_model.timing_enabled = True
        g_model.dT = dt
        g_model._model.set_seed(rng_seed)

        # Add input neurons
        n = self.weight_vals[0].shape[0]
        input_type = InputType(input_type)
        if input_type == InputType.IF:
            nrn_post = [g_model.add_neuron_population(
                'input_nrn_' + str(batch_i), n, if_input_model, {}, if_input_init
            ) for batch_i in range(batch_size)]
        elif input_type == InputType.POISSON:
            nrn_post = [g_model.add_neuron_population(
                'input_nrn_' + str(batch_i), n, poisson_input_model, {'rate_factor': rate_factor}, poisson_input_init
            ) for batch_i in range(batch_size)]
        elif input_type == InputType.SPIKE:
            nrn_post = [g_model.add_neuron_population(
                'input_nrn_' + str(batch_i), n, spike_input_model, {}, spike_input_init
            ) for batch_i in range(batch_size)]

        # For each synapse population
        for name, w_vals, w_conn, thr in zip(self.layer_names, self.weight_vals, self.weight_conn, self.thresholds):
            nrn_pre = nrn_post

            # Add next layer of neurons
            n = w_vals.shape[1]
            nrn_post = [g_model.add_neuron_population(
                name + '_nrn_' + str(batch_i), n, if_model, {}, if_init
            ) for batch_i in range(batch_size)]
            for nrn_post_i in nrn_post:
                nrn_post_i.set_extra_global_param('Vthr', thr)

            # Add synapses from last layer to this layer
            if w_conn.all(): # Dense weight matrix
                syn = [g_model.add_synapse_population(
                    name + '_syn_' + str(batch_i), 'DENSE_INDIVIDUALG', genn_wrapper.NO_DELAY, nrn_pre[batch_i],
                    nrn_post[batch_i], 'StaticPulse', {}, {'g': w_vals.flatten()}, {}, {}, 'DeltaCurr', {}, {}
                ) for batch_i in range(batch_size)]
            else: # Sparse weight matrix
                w_inds = np.nonzero(w_conn)
                syn = [g_model.add_synapse_population(
                    name + '_syn_' + str(batch_i), 'SPARSE_INDIVIDUALG', genn_wrapper.NO_DELAY, nrn_pre[batch_i],
                    nrn_post[batch_i], 'StaticPulse', {}, {'g': w_vals[w_inds]}, {}, {}, 'DeltaCurr', {}, {}
                ) for batch_i in range(batch_size)]
                for syn_i in syn:
                    syn_i.set_sparse_connections(w_inds[0], w_inds[1])

        # Build and load model
        self.g_model = g_model
        self.g_model.build()
        self.g_model.load()


    def set_input_batch(self, x_data):
        """Set model input with a new batch of samples

        Args:
        x_data  --  data used to set model inputs with
        """

        # Input sanity check
        n_samples = x_data.shape[0]
        sample_size = np.prod(x_data.shape[1:])
        input_size = self.weight_vals[0].shape[0]
        if x_data.shape[0] > self.batch_size:
            raise ValueError('sample count {} > batch size {}'.format(n_samples, self.batch_size))
        if sample_size != self.weight_vals[0].shape[0]:
            raise ValueError('sample size {} != input size {}'.format(sample_size, input_size))

        # Set model inputs
        for i in range(self.batch_size):
            input_name = 'input_nrn_' + str(i)
            input_nrn = self.g_model.neuron_populations[input_name]
            if i < n_samples:
                input_nrn.vars['input'].view[:] = x_data[i].flatten()
            else:
                input_nrn.vars['input'].view[:] = np.zeros(input_size)
            self.g_model.push_state_to_device(input_name)


    def step_time(self, iterations=1):
        """Iterate the GeNN model a given number of steps

        Keyword args:
        iterations  --  number of iterations (default: 1)
        """

        for i in range(iterations):
            self.g_model.step_time()


    def reset_state(self):
        """Reset the GeNN model's state to initial values"""

        self.g_model._slm.initialize()
        self.g_model.timestep = 0
        self.g_model.t = 0.0


    def evaluate(self, x_data, y_data, time, save_samples=[]):
        """Evaluate the accuracy of a GeNN model

        Args:
        x_data        --  test samples
        y_data        --  test labels
        time          --  sample present time (msec)

        Keyword args:
        save_samples  --  list of sample indices to save spikes for (default: [])

        Returns:
        accuracy      --  percentage of correctly classified results
        spike_i       --  list of spike indices for each sample index in save_samples
        spike_t       --  list of spike times for each sample index in save_samples
        """

        # Input sanity check
        n_samples = x_data.shape[0]
        n_labels = y_data.shape[0]
        if n_samples != n_labels:
            raise ValueError('sample count {} != label count {}'.format(n_samples, n_labels))

        n_correct = 0
        spike_i = [[np.empty(0)] * len(self.g_model.neuron_populations)] * len(save_samples)
        spike_t = [[np.empty(0)] * len(self.g_model.neuron_populations)] * len(save_samples)

        # For each sample batch
        progress = tqdm(total=n_samples)
        for batch_start in range(0, n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_samples)
            batch_x_data = x_data[batch_start:batch_end]
            batch_y_data = y_data[batch_start:batch_end]

            # Set new input
            self.reset_state()
            self.set_input_batch(batch_x_data)

            # Main simulation loop
            while self.g_model.t < time:

                # Step time
                self.step_time()

                # Save spikes
                for sample_i in [i for i in save_samples if batch_start <= i < batch_end]:
                    k = save_samples.index(sample_i)
                    batch_i = sample_i - batch_start
                    names = ['input_nrn_' + str(batch_i)]
                    names += [name + '_nrn_' + str(batch_i) for name in self.layer_names]
                    for l, name in enumerate(names):
                        nrn = self.g_model.neuron_populations[name]
                        self.g_model.pull_current_spikes_from_device(name)
                        indices = nrn.current_spikes
                        times = np.ones(indices.shape) * self.g_model.t
                        spike_i[k][l] = np.hstack((spike_i[k][l], indices))
                        spike_t[k][l] = np.hstack((spike_t[k][l], times))

            # After simulation
            for batch_i in range(batch_end - batch_start):
                output_name = self.layer_names[-1] + '_nrn_' + str(batch_i)
                output_nrn = self.g_model.neuron_populations[output_name]
                self.g_model.pull_var_from_device(output_name, 'nSpk')
                n_correct += output_nrn.vars['nSpk'].view.argmax() == batch_y_data[batch_i]

            accuracy = (n_correct / batch_end) * 100
            progress.set_postfix_str('accuracy: {:2.2f}'.format(accuracy))
            progress.update(batch_end - batch_start)

        progress.close()

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
