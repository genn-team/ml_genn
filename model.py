
import math

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
        self.weight_inds = None
        self.weight_vals = None

    def create_genn_model(self, dt=1.0, rng_seed=0, rate_factor=1.0):
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
        self.weight_inds = []
        self.weight_vals = []

        # For each TensorFlow model layer:
        for layer in self.tf_model.layers:

            # === Flatten layers ===
            if isinstance(layer, tf.keras.layers.Flatten):
                continue

            # === Dense layers ===
            elif isinstance(layer, tf.keras.layers.Dense):
                tf_w = layer.get_weights()[0]
                g_w = tf_w.copy()

                self.layer_names.append(layer.name)
                self.weight_inds.append(None)
                self.weight_vals.append(g_w)
                g_w[np.isnan(g_w)] = 0.0

            # === Conv2D layers ===
            elif isinstance(layer, tf.keras.layers.Conv2D):
                tf_w = layer.get_weights()[0]
                g_w = np.full((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])), np.nan)

                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # Stride centre point iterators.
                if layer.padding == 'valid':
                    stride_rows = range(0, ih - kh + 1, sh)
                    stride_cols = range(0, iw - kw + 1, sw)
                elif layer.padding == 'same':
                    stride_rows = range(0 - kh // 2, ih - kh // 2, sh)
                    stride_cols = range(0 - kw // 2, iw - kw // 2, sw)

                # For each input channel -> output channel kernel:
                for in_channel in range(ic):
                    for out_channel in range(oc):
                        tf_kernel = tf_w[:, :, in_channel, out_channel]
                        g_kernel = g_w[in_channel::ic, out_channel::oc]

                        # For each output neuron:
                        for out_row, stride_row in enumerate(stride_rows):
                            for out_col, stride_col in enumerate(stride_cols):
                                g_out = g_kernel[:, out_row * ow + out_col]
                                g_out.shape = (ih, iw)

                                # Get a kernel stride view in tf_w.
                                tf_kernel_T = 0 - min(stride_row, 0)
                                tf_kernel_B = kh - max(stride_row + kh - ih, 0)
                                tf_kernel_L = 0 - min(stride_col, 0)
                                tf_kernel_R = kw - max(stride_col + kw - iw, 0)
                                tf_stride = tf_kernel[tf_kernel_T:tf_kernel_B, tf_kernel_L:tf_kernel_R]

                                # Get a kernel stride view in g_w.
                                g_kernel_T = max(stride_row, 0)
                                g_kernel_B = min(stride_row + kh, ih)
                                g_kernel_L = max(stride_col, 0)
                                g_kernel_R = min(stride_col + kw, iw)
                                g_stride = g_out[g_kernel_T:g_kernel_B, g_kernel_L:g_kernel_R]

                                # Set weights for this stride.
                                g_stride[:] = tf_stride

                self.layer_names.append(layer.name)
                self.weight_inds.append(np.nonzero(~np.isnan(g_w)))
                self.weight_vals.append(g_w)
                g_w[np.isnan(g_w)] = 0.0

            # === AveragePooling2D layers ===
            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                g_w = np.zeros((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])))

                ph, pw = layer.pool_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # Stride centre point iterators.
                if layer.padding == 'valid':
                    stride_rows = range(0, ih - ph + 1, sh)
                    stride_cols = range(0, iw - pw + 1, sw)
                elif layer.padding == 'same':
                    stride_rows = range(0 - ph // 2, ih - ph // 2, sh)
                    stride_cols = range(0 - pw // 2, iw - pw // 2, sw)

                # For each channel pool:
                for channel in range(ic):
                    g_pool = g_w[channel::ic, channel::ic]

                    # For each output neuron:
                    for out_row, stride_row in enumerate(stride_rows):
                        for out_col, stride_col in enumerate(stride_cols):
                            g_out = g_pool[:, out_row * ow + out_col]
                            g_out.shape = (ih, iw)

                            # Get a pool stride view in g_w.
                            g_pool_T = max(stride_row, 0)
                            g_pool_B = min(stride_row + ph, ih)
                            g_pool_L = max(stride_col, 0)
                            g_pool_R = min(stride_col + pw, iw)
                            g_stride = g_out[g_pool_T:g_pool_B, g_pool_L:g_pool_R]

                            # Set weights for this stride.
                            g_stride[:] = 1.0 / (ph * pw)

                self.layer_names.append(layer.name)
                self.weight_inds.append(np.nonzero(~np.isnan(g_w)))
                self.weight_vals.append(g_w)
                g_w[np.isnan(g_w)] = 0.0


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
            extra_global_params=[('factor', 'scalar')],
            var_name_types=[('rate', 'scalar')],
            threshold_condition_code='''
            $(gennrand_uniform) >= exp(-$(rate) * $(factor) * DT)
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
        self.g_model = genn_model.GeNNModel('float', 'tg_model')
        self.g_model.dT = dt
        self.g_model._model.set_seed(rng_seed)

        # Add Poisson distributed spiking inputs
        n = np.prod(self.tf_model.input_shape[1:])
        nrn_post = self.g_model.add_neuron_population('input_nrn', n, poisson_model, {}, poisson_init)
        nrn_post.set_extra_global_param('factor', rate_factor)

        # # Add inputs with constant injected current
        # n = np.prod(self.tf_model.input_shape[1:])
        # nrn_post = self.g_model.add_neuron_population('input_nrn', n, if_model, if_params, if_init)
        # nrn_post.set_extra_global_param('Vthr', 1.0)
        # self.g_model.add_current_source('input_cs', cs_model, 'input_nrn', {}, cs_init)

        # For each synapse population
        for name, w_inds, w_vals in zip(self.layer_names, self.weight_inds, self.weight_vals):
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


    def evaluate_genn_model(self, x, y, present_time=100.0, save_samples=[]):
        n_correct = 0
        n_examples = len(x)

        spike_idx = [[None] * len(self.g_model.neuron_populations)] * len(save_samples)
        spike_times = [[None] * len(self.g_model.neuron_populations)] * len(save_samples)

        # For each sample presentation
        for i in range(n_examples):

            # Before simulation
            for ln in self.layer_names:
                nrn = self.g_model.neuron_populations[ln + '_nrn']
                nrn.vars['Vmem'].view[:] = 0.0
                nrn.vars['Vmem_peak'].view[:] = 0.0
                nrn.vars['nSpk'].view[:] = 0
                self.g_model.push_state_to_device(ln + '_nrn')

            # === Poisson inputs
            nrn = self.g_model.neuron_populations['input_nrn']
            nrn.vars['rate'].view[:] = x[i].flatten()
            self.g_model.push_state_to_device('input_nrn')

            # # === IF inputs with constant current
            # nrn = self.g_model.neuron_populations['input_nrn']
            # nrn.vars['Vmem'].view[:] = 0.0
            # nrn.vars['Vmem_peak'].view[:] = 0.0
            # nrn.vars['nSpk'].view[:] = 0
            # self.g_model.push_state_to_device('input_nrn')
            # cs = self.g_model.current_sources['input_cs']
            # cs.vars['magnitude'].view[:] = x[i].flatten()
            # self.g_model.push_state_to_device('input_cs')

            # Run simulation
            for t in range(math.ceil(present_time / self.g_model.dT)):
                self.g_model.step_time()

                # Save spikes
                if i in save_samples:
                    k = save_samples.index(i)
                    names = ['input_nrn'] + [name + '_nrn' for name in self.layer_names]
                    neurons = [self.g_model.neuron_populations[name] for name in names]
                    for j, npop in enumerate(neurons):
                        self.g_model.pull_current_spikes_from_device(npop.name)
                        idx = npop.current_spikes    # size of npop
                        ts = np.ones(idx.shape) * t  # size of npop
                        if spike_idx[k][j] is None:
                            spike_idx[k][j] = np.copy(idx)
                            spike_times[k][j] = ts
                        else:
                            spike_idx[k][j] = np.hstack((spike_idx[k][j], idx))
                            spike_times[k][j] = np.hstack((spike_times[k][j], ts))

            # After simulation
            output_neurons = self.g_model.neuron_populations[self.layer_names[-1] + '_nrn']
            self.g_model.pull_var_from_device(output_neurons.name, 'nSpk')
            nSpk_view = output_neurons.vars['nSpk'].view
            n_correct += (np.argmax(nSpk_view) == y[i])

        accuracy = (n_correct / n_examples) * 100.

        return accuracy, spike_idx, spike_times
