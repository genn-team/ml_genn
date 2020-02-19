from math import ceil
import numpy as np
import tensorflow as tf
from pygenn import genn_model, genn_wrapper

supported_layers = (
    tf.keras.layers.Dense,
    tf.keras.layers.Conv2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.Flatten,
)

class TGModel():
    def __init__(self, tf_model=None, g_model=None):
        self.tf_model = tf_model
        self.g_model = g_model
        self.layer_names = None
        self.weight_vals = None
        self.weight_inds = None

    def create_genn_model(self, dt=1.0, rng_seed=0, rate_factor=1.0, input_type='poisson'):
        # Check model compatibility
        if not isinstance(self.tf_model, tf.keras.Sequential):
            raise NotImplementedError('{} models not supported'.format(type(self.tf_model)))
        for layer in self.tf_model.layers[:-1]:
            if not isinstance(layer, supported_layers):
                raise NotImplementedError('{} layers are not supported'.format(layer))
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.activation != tf.keras.activations.relu:
                    raise NotImplementedError('{} activation is not supported'.format(layer.activation))
                if layer.use_bias == True:
                    raise NotImplementedError('bias tensors are not supported')

        self.layer_names = []
        self.weight_vals = []
        self.weight_inds = []
        deferred_w = None
        deferred_conn = None

        # For each TensorFlow model layer:
        for layer in self.tf_model.layers:

            # === Flatten Layers ===
            if isinstance(layer, tf.keras.layers.Flatten):
                continue

            # === Dense Layers ===
            elif isinstance(layer, tf.keras.layers.Dense):
                tf_w = layer.get_weights()[0]
                g_w = tf_w.copy()
                g_conn = np.ones(g_w.shape, dtype=np.bool)
                defer_weights = False

            # === Conv2D Layers ===
            elif isinstance(layer, tf.keras.layers.Conv2D):
                tf_w = layer.get_weights()[0]
                g_w = np.zeros((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])))
                g_conn = np.zeros(g_w.shape, dtype=np.bool)
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

                # For each input -> output kernel:
                for in_channel in range(ic):
                    for out_channel in range(oc):
                        tf_channel_w = tf_w[:, :, in_channel, out_channel]
                        g_channel_w = g_w[in_channel::ic, out_channel::oc]
                        g_channel_conn = g_conn[in_channel::ic, out_channel::oc]

                        # For each output neuron:
                        for out_row, stride_row in enumerate(stride_rows):
                            for out_col, stride_col in enumerate(stride_cols):
                                g_out_w = g_channel_w[:, out_row * ow + out_col]
                                g_out_w.shape = (ih, iw)
                                g_out_conn = g_channel_conn[:, out_row * ow + out_col]
                                g_out_conn.shape = (ih, iw)

                                # Get a kernel stride view in tf_channel_w.
                                tf_kern_T = 0 - min(stride_row, 0)
                                tf_kern_B = kh - max(stride_row + kh - ih, 0)
                                tf_kern_L = 0 - min(stride_col, 0)
                                tf_kern_R = kw - max(stride_col + kw - iw, 0)
                                tf_stride_w = tf_channel_w[tf_kern_T:tf_kern_B, tf_kern_L:tf_kern_R]

                                # Get a kernel stride view in g_out_w.
                                g_kern_T = max(stride_row, 0)
                                g_kern_B = min(stride_row + kh, ih)
                                g_kern_L = max(stride_col, 0)
                                g_kern_R = min(stride_col + kw, iw)
                                g_stride_w = g_out_w[g_kern_T:g_kern_B, g_kern_L:g_kern_R]
                                g_stride_conn = g_out_conn[g_kern_T:g_kern_B, g_kern_L:g_kern_R]

                                # Set weights for this stride.
                                g_stride_w[:] = tf_stride_w
                                g_stride_conn[:] = True

            # === AveragePooling2D Layers ===
            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                g_w = np.zeros((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])))
                g_conn = np.zeros(g_w.shape, dtype=np.bool)
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

                # For each input -> output pool:
                for channel in range(ic):
                    g_pool_w = g_w[channel::ic, channel::ic]
                    g_pool_conn = g_conn[channel::ic, channel::ic]

                    # For each output neuron:
                    for out_row, stride_row in enumerate(stride_rows):
                        for out_col, stride_col in enumerate(stride_cols):
                            g_out_w = g_pool_w[:, out_row * ow + out_col]
                            g_out_w.shape = (ih, iw)
                            g_out_conn = g_pool_conn[:, out_row * ow + out_col]
                            g_out_conn.shape = (ih, iw)

                            # Get a pool stride view in g_out_w.
                            g_pool_T = max(stride_row, 0)
                            g_pool_B = min(stride_row + ph, ih)
                            g_pool_L = max(stride_col, 0)
                            g_pool_R = min(stride_col + pw, iw)
                            g_stride_w = g_out_w[g_pool_T:g_pool_B, g_pool_L:g_pool_R]
                            g_stride_conn = g_out_conn[g_pool_T:g_pool_B, g_pool_L:g_pool_R]

                            # Set weights for this stride.
                            g_stride_w[:] = 1.0 / (ph * pw)
                            g_stride_conn[:] = True

            # === Combine Deferred Weights ===
            if deferred_w is not None:
                new_w = np.empty((deferred_w.shape[0], g_w.shape[1]))
                new_conn = np.empty(new_w.shape, dtype=np.bool)

                # For each input -> output weight matrix:
                for in_channel in range(ic):
                    # Note to future devs: deferred_* indexing below assumes that the deferred
                    # weight matrix has an equal number of input and output channels, and maps
                    # input channel i one-to-one to output channel i, for all i in [0, ic).
                    deferred_channel_w = deferred_w[in_channel::ic, in_channel::ic]
                    deferred_channel_conn = deferred_conn[in_channel::ic, in_channel::ic]
                    for out_channel in range(oc):
                        g_channel_w = g_w[in_channel::ic, out_channel::oc]
                        g_channel_conn = g_conn[in_channel::ic, out_channel::oc]
                        new_channel_w = new_w[in_channel::ic, out_channel::oc]
                        new_channel_conn = new_conn[in_channel::ic, out_channel::oc]

                        # Set weights to dot product of deferred and new weights.
                        new_channel_w[:] = np.dot(deferred_channel_w, g_channel_w)
                        new_channel_conn[:] = np.dot(deferred_channel_conn, g_channel_conn)

                # Update weights.
                g_w = new_w
                g_conn = new_conn

            # === Append Weights to Model ===
            if defer_weights:
                # Defer weights to next layer.
                deferred_w = g_w
                deferred_conn = g_conn
            else:
                # Append weights for this layer.
                deferred_w = None
                deferred_conn = None
                self.layer_names.append(layer.name)
                self.weight_vals.append(g_w)
                if g_conn.all():
                    self.weight_inds.append(None)
                else:
                    self.weight_inds.append(np.nonzero(g_conn))


        # === Define IF neuron class ===
        if_model = genn_model.create_custom_neuron_class(
            'if_model',
            param_names=['Vres'],
            extra_global_params=[('Vthr', 'scalar')],
            var_name_types=[('Vmem', 'scalar'), ('Vmem_peak', 'scalar'), ('nSpk', 'unsigned int')],
            sim_code='''
            $(Vmem) += $(Isyn) * DT;
            $(Vmem_peak) = $(Vmem);
            ''',
            reset_code='''
            $(Vmem) = $(Vres);
            $(nSpk) += 1;
            ''',
            threshold_condition_code='''
            $(Vmem) >= $(Vthr)
            '''
        )

        if_params = {
            'Vres': 0.0,
        }

        if_init = {
            'Vmem': 0.0,
            'Vmem_peak': 0.0,
            'nSpk': 0,
        }

        # === Define Poisson neuron class ===
        poisson_model = genn_model.create_custom_neuron_class(
            'poisson_model',
            extra_global_params=[('rate_factor', 'scalar')],
            var_name_types=[('rate', 'scalar')],
            threshold_condition_code='''
            $(gennrand_uniform) >= exp(-$(rate) * $(rate_factor) * DT)
            '''
        )

        poisson_init = {
            'rate': 0.0,
        }

        # === Define current source class ===
        cs_model = genn_model.create_custom_current_source_class(
            'cs_model',
            var_name_types=[('magnitude', 'scalar')],
            injection_code='''
            $(injectCurrent, $(magnitude));
            '''
        )

        cs_init = {
            'magnitude': 0.0,
        }


        # Define GeNN model
        self.g_model = genn_model.GeNNModel('float', self.tf_model.name)
        self.g_model.dT = dt
        self.g_model._model.set_seed(rng_seed)

        if input_type == 'if_cs':
            # Add inputs with constant injected current
            n = np.prod(self.tf_model.input_shape[1:])
            nrn_post = self.g_model.add_neuron_population('input_nrn', n, if_model, if_params, if_init)
            nrn_post.set_extra_global_param('Vthr', 1.0)
            self.g_model.add_current_source('input_cs', cs_model, 'input_nrn', {}, cs_init)
        elif input_type == 'poisson':
            # Add Poisson distributed spiking inputs
            n = np.prod(self.tf_model.input_shape[1:])
            nrn_post = self.g_model.add_neuron_population('input_nrn', n, poisson_model, {}, poisson_init)
            nrn_post.set_extra_global_param('rate_factor', rate_factor)

        # For each synapse population
        for name, w_vals, w_inds in zip(self.layer_names, self.weight_vals, self.weight_inds):
            nrn_pre = nrn_post

            # Add next layer of neurons
            n = w_vals.shape[1]
            nrn_post = self.g_model.add_neuron_population(name + '_nrn', n, if_model, if_params, if_init)
            nrn_post.set_extra_global_param('Vthr', 1.0)

            # Add synapses from last layer to this layer
            if w_inds is None: # Dense weight matrix
                syn = self.g_model.add_synapse_population(
                    name + '_syn', 'DENSE_INDIVIDUALG', genn_wrapper.NO_DELAY, nrn_pre, nrn_post,
                    'StaticPulse', {}, {'g': w_vals.flatten()}, {}, {}, 'DeltaCurr', {}, {}
                )
            else: # Sparse weight matrix
                syn = self.g_model.add_synapse_population(
                    name + '_syn', 'SPARSE_INDIVIDUALG', genn_wrapper.NO_DELAY, nrn_pre, nrn_post,
                    'StaticPulse', {}, {'g': w_vals[w_inds]}, {}, {}, 'DeltaCurr', {}, {}
                )
                syn.set_sparse_connections(w_inds[0], w_inds[1])

        # Build and load model
        self.g_model.build()
        self.g_model.load()

    def reset_state(self):
        self.g_model._slm.initialize()
        self.g_model._slm.set_timestep(0)
        self.g_model._slm.set_time(0.0)

    def set_inputs(self, x):
        if self.g_model.current_sources.get('input_cs') is not None:
            # IF inputs with constant current
            cs = self.g_model.current_sources['input_cs']
            cs.vars['magnitude'].view[:] = x.flatten()
            self.g_model.push_state_to_device('input_cs')
        else:
            # Poisson inputs
            nrn = self.g_model.neuron_populations['input_nrn']
            nrn.vars['rate'].view[:] = x.flatten()
            self.g_model.push_state_to_device('input_nrn')

    def step_time(self, iterations=1):
        for i in range(iterations):
            self.g_model.step_time()

    def evaluate_genn_model(self, x_data, y_data, save_samples=[], classify_time=500.0, classify_spikes=None):
        assert x_data.shape[0] == y_data.shape[0]
        n_correct = 0

        spike_idx = [[None] * len(self.g_model.neuron_populations)] * len(save_samples)
        spike_times = [[None] * len(self.g_model.neuron_populations)] * len(save_samples)

        # For each sample presentation
        for i, (x, y) in enumerate(zip(x_data, y_data)):

            # Reset state
            self.reset_state()

            # Set inputs
            self.set_inputs(x)

            # Main simulation loop
            while self.g_model.t < classify_time:
                t = self.g_model.t

                # Step time
                self.step_time()

                # Save spikes
                if i in save_samples:
                    k = save_samples.index(i)
                    names = ['input_nrn'] + [name + '_nrn' for name in self.layer_names]
                    neurons = [self.g_model.neuron_populations[name] for name in names]
                    for j, nrn in enumerate(neurons):
                        self.g_model.pull_current_spikes_from_device(nrn.name)
                        idx = nrn.current_spikes
                        ts = np.ones(idx.shape) * t
                        if spike_idx[k][j] is None:
                            spike_idx[k][j] = np.copy(idx)
                            spike_times[k][j] = ts
                        else:
                            spike_idx[k][j] = np.hstack((spike_idx[k][j], idx))
                            spike_times[k][j] = np.hstack((spike_times[k][j], ts))

                # Break simulation if we have enough output spikes.
                if classify_spikes is not None:
                    output_neurons = self.g_model.neuron_populations[self.layer_names[-1] + '_nrn']
                    self.g_model.pull_var_from_device(output_neurons.name, 'nSpk')
                    if output_neurons.vars['nSpk'].view.sum() >= classify_spikes:
                        break

            # After simulation
            output_neurons = self.g_model.neuron_populations[self.layer_names[-1] + '_nrn']
            self.g_model.pull_var_from_device(output_neurons.name, 'nSpk')
            n_correct += output_neurons.vars['nSpk'].view.argmax() == y

        accuracy = (n_correct / len(x_data)) * 100.

        return accuracy, spike_idx, spike_times
