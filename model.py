
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
        self.weight_vals = None
        self.weight_inds = None

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
        self.weight_vals = []
        self.weight_inds = []

        for layer in self.tf_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                tf_w = layer.get_weights()[0]
                g_w = tf_w.copy()

                self.layer_names.append(layer.name)
                self.weight_vals.append(g_w)
                self.weight_inds.append(None)

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
                        # Get tf_w and g_w views for this kernel.
                        tf_k = tf_w[:, :, in_channel, out_channel]
                        g_k = g_w[in_channel::ic, out_channel::oc]

                        # For each kernel stride:
                        for out_row, stride_row in enumerate(stride_rows):
                            for out_col, stride_col in enumerate(stride_cols):

                                # Get kernel boundaries for this stride in tf_w.
                                tf_k_row_T = 0 - min(stride_row, 0)
                                tf_k_row_B = kh - max(stride_row + kh - ih, 0)
                                tf_k_col_L = 0 - min(stride_col, 0)
                                tf_k_col_R = kw - max(stride_col + kw - iw, 0)

                                # Get kernel boundaries for this stride in g_w.
                                g_k_row_T = max(stride_row, 0)
                                g_k_row_B = min(stride_row + kh, ih)
                                g_k_col_L = max(stride_col, 0)
                                g_k_col_R = min(stride_col + kw, iw)

                                # Set weights for this stride.
                                tf_stride = tf_k[tf_k_row_T:tf_k_row_B, tf_k_col_L:tf_k_col_R]
                                g_stride = g_k[:, out_row * ow + out_col]
                                g_stride.shape = (ih, iw)
                                g_stride[g_k_row_T:g_k_row_B, g_k_col_L:g_k_col_R] = tf_stride

                self.layer_names.append(layer.name)
                self.weight_vals.append(g_w)
                self.weight_inds.append(np.nonzero(~np.isnan(g_w)))

            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                g_w = np.full((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])), np.nan)

                ph, pw = layer.pool_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                for n in range(ow * oh): # output unit index
                    for k in range(pw): # over kernel width
                        for l in range(ph): # over kernel height
                            # 1.0 / (kernel size) is the weight from input neurons to corresponding output neurons
                            g_w[(n % ow) * ic * sh + (n // ow) * ic * iw * sw + k * ic * iw + l * ic:
                                (n % ow) * ic * sh + (n // ow) * ic * iw * sw + k * ic * iw + l * ic + ic,
                                n * oc : n * oc + oc] = np.diag([1.0 / (pw * ph)] * oc) # diag since we need a one-to-one mapping along channels

                self.layer_names.append(layer.name)
                self.weight_vals.append(g_w)
                self.weight_inds.append(np.nonzero(~np.isnan(g_w)))

            elif isinstance(layer, tf.keras.layers.Flatten):
                # Ignore Flatten layers
                continue


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
                w_vals = w_vals[w_inds]
                syn = self.g_model.add_synapse_population(
                    name + '_syn', 'SPARSE_INDIVIDUALG', genn_wrapper.NO_DELAY, nrn_pre, nrn_post,
                    'StaticPulse', {}, {'g': w_vals.copy()}, {}, {}, 'DeltaCurr', {}, {}
                )
                syn.set_sparse_connections(w_inds[0].copy(), w_inds[1].copy())

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
