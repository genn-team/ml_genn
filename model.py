
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

    def create_genn_model(self, dt=1.0):
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
                genn_w = tf_w.copy()

                self.layer_names.append(layer.name)
                self.weight_vals.append(genn_w)
                self.weight_inds.append(None)

            elif isinstance(layer, tf.keras.layers.Conv2D):
                tf_w = layer.get_weights()[0]
                genn_w = np.full((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])), np.nan)

                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # No padding, such that output.shape < input.shape.
                if layer.padding == 'valid':
                    # oh = ceil((ih - kh + 1) / sh)
                    # ow = ceil((iw - kw + 1) / sw)

                    # For each post-synapse neuron (each column in genn_w):
                    for i_post in range(genn_w.shape[1]):

                        # Find its output channel index and output pixel coordinates.
                        out_channel    = i_post % oc
                        out_pixel_flat = i_post // oc
                        out_pixel_row  = out_pixel_flat // ow
                        out_pixel_col  = out_pixel_flat % ow

                        # Find the row (pre-synapse neuron) offset of genn_w corresponding to the input pixel
                        # at the first row and first column of the convolution kernel's current stride.
                        in_row_offset = out_pixel_row * sh * ic * iw    # input row offset
                        in_col_offset = out_pixel_col * sw * ic         # input col offset
                        pre_offset    = in_row_offset + in_col_offset

                        # For each row of the kernels that filter each input channel into this output channel,
                        # place that row (all input channels) into genn_w at each kernel row block offset.
                        for k_row in range(kh):
                            k_row_start = pre_offset + k_row * ic * iw  # kernel row start in genn_w
                            k_row_end   = k_row_start + ic * kw         # kernel row end in genn_w
                            k_row_wt    = tf_w[k_row, :, :, out_channel]
                            genn_w[k_row_start : k_row_end, i_post] = k_row_wt.flatten()

                # Zero padding, such that output.shape == input.shape.
                elif layer.padding == 'same':
                    # oh = ceil(ih / sh)
                    # ow = ceil(iw / sw)

                    # TODO: strides are NOT handled correctly yet

                    # Calculate padding for each side of input map, as done by tf and keras.
                    # https://www.tensorflow.org/versions/r1.12/api_guides/python/nn
                    if (ih % sh == 0):
                        pad_along_height = max(kh - sh,        0)
                    else:
                        pad_along_height = max(kh - (ih % sh), 0)
                    if (iw % sw == 0):
                        pad_along_width  = max(kw - sw,        0)
                    else:
                        pad_along_width  = max(kw - (iw % sw), 0)

                    pad_top    = pad_along_height // 2
                    pad_bottom = pad_along_height - pad_top
                    pad_left   = pad_along_width // 2
                    pad_right  = pad_along_width - pad_left

                    # For each post-synapse neuron (each column in genn_w):
                    for i_post in range(genn_w.shape[1]):

                        # Find its output channel index and output pixel coordinates.
                        out_channel    = i_post % oc
                        out_pixel_flat = i_post // oc
                        out_pixel_row  = out_pixel_flat // ow
                        out_pixel_col  = out_pixel_flat % ow

                        # Calculate effective start and end indices on input matrix, along x and y dimensions
                        starth = max(out_pixel_row - pad_top,    0)      * sh * ic * iw
                        endh   = min(out_pixel_row + pad_bottom, ih - 1) * sh * ic * iw
                        startw = max(out_pixel_col - pad_left,   0)      * sw * ic
                        endw   = min(out_pixel_col + pad_right,  iw - 1) * sw * ic + ic

                        # Calculate start and end indices for weight matrix to be assigned at the synapse
                        startkw = max(pad_left - ((i_post * sw) // oc) % ow, 0)
                        endkw   = startkw + (endw - ic - startw) // ic + 1
                        startkh = max(pad_top - ((i_post * sh) // oc) // ow, 0)

                        # Weight mapping
                        for k in range(starth, endh+1, ic*iw):
                            w = tf_w[startkh + (k-starth)//(ic*iw), startkw : endkw, :, out_channel]
                            genn_w[k+startw : k+endw, i_post] = w.flatten()

                self.layer_names.append(layer.name)
                self.weight_vals.append(genn_w)
                self.weight_inds.append(np.nonzero(~np.isnan(genn_w)))

            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                genn_w = np.full((np.prod(layer.input_shape[1:]), np.prod(layer.output_shape[1:])), np.nan)

                ph, pw = layer.pool_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                for n in range(ow * oh): # output unit index
                    for k in range(pw): # over kernel width
                        for l in range(ph): # over kernel height
                            # 1.0 / (kernel size) is the weight from input neurons to corresponding output neurons
                            genn_w[(n % ow) * ic * sh + (n // ow) * ic * iw * sw + k * ic * iw + l * ic:
                                   (n % ow) * ic * sh + (n // ow) * ic * iw * sw + k * ic * iw + l * ic + ic,
                                   n * oc : n * oc + oc] = np.diag([1.0 / (pw * ph)] * oc) # diag since we need a one-to-one mapping along channels

                self.layer_names.append(layer.name)
                self.weight_vals.append(genn_w)
                self.weight_inds.append(np.nonzero(~np.isnan(genn_w)))

            elif isinstance(layer, tf.keras.layers.Flatten):
                # Ignore Flatten layers
                continue


        # Define IF neuron class
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

        # Define current source class
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

        # Define Poisson neuron class
        poisson_model = genn_model.create_custom_neuron_class(
            'poisson_model',
            extra_global_params=[('rate', 'scalar')],
            threshold_condition_code='''
            $(gennrand_uniform) >= exp(-$(rate) * DT)
            '''
        )


        # Define GeNN model
        self.g_model = genn_model.GeNNModel('float', 'tg_model')
        self.g_model.dT = dt



        # ========== INPUTS AS POISSON NEURONS
        # Add Poisson distributed spiking inputs
        n = np.prod(self.tf_model.input_shape[1:])
        nrn_post = self.g_model.add_neuron_population('input_nrn', n, poisson_model, {}, {})
        nrn_post.set_extra_global_param('rate', 0.0)






        # # ========== INPUTS WITH INJECTED CURRENT
        # # Add inputs with constant injected current
        # n = np.prod(self.tf_model.input_shape[1:])
        # nrn_post = self.g_model.add_neuron_population('input_nrn', n, if_model, if_params, if_init)
        # nrn_post.set_extra_global_param('Vthr', 1.0)
        # self.g_model.add_current_source(
        #     'input_cs', cs_model, 'input_nrn', {}, cs_init
        # )




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


    def evaluate_genn_model(self, x, y, present_time=100.0, rate_factor=1.0, save_samples=[]):
        n_correct = 0
        n_examples = len(x)

        spike_idx = [[None] * len(self.g_model.neuron_populations)] * len(save_samples)
        spike_times = [[None] * len(self.g_model.neuron_populations)] * len(save_samples)

        # For each sample presentation
        for i in range(n_examples):

            # Before simulation
            for name in self.layer_names:
                nrn = self.g_model.neuron_populations[name + '_nrn']
                nrn.vars['Vmem'].view[:] = 0.0
                nrn.vars['Vmem_peak'].view[:] = 0.0
                nrn.vars['nSpk'].view[:] = 0
                self.g_model.push_state_to_device(nrn.name)



            # ========== SET RATE AS INPUT * RATE_FACTOR



            # # Before simulation
            # for nrn in self.g_model.neuron_populations.values():
            #     nrn.vars['Vmem'].view[:] = 0.0
            #     nrn.vars['Vmem_peak'].view[:] = 0.0
            #     nrn.vars['nSpk'].view[:] = 0
            #     self.g_model.push_state_to_device(nrn.name)

            # # TODO: INPUT RATE ENCODING
            # # FOR NOW, USE CONSTANT CURRENT INJECTION EQUAL TO INPUT MAGNITUDE
            # self.g_model.current_sources['input_cs'].vars['magnitude'].view[:] = x[i].flatten()
            # self.g_model.push_var_to_device('input_cs', 'magnitude')


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
