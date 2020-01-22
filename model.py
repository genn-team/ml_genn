
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
    def __init__(self, tf_model=None, genn_model=None, dt=1.0):
        # Check model compatibility
        if not isinstance(tf_model, tf.keras.Sequential):
            raise NotImplementedError('{} models not supported'.format(type(tf_model)))
        for layer in tf_model.layers[:-1]:
            if not isinstance(layer, supported_layers):
                raise NotImplementedError('{} layers are not supported'.format(layer))
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.activation != tf.keras.activations.relu:
                    raise NotImplementedError('{} activation is not supported'.format(layer.activation))
                if layer.use_bias == True:
                    raise NotImplementedError('bias tensors are not supported')

        self.tf_model = tf_model
        self.genn_model = genn_model
        self.genn_n_neurons = None
        self.genn_w_name = None
        self.genn_w_inds = None
        self.genn_w_vals = None
        self.genn_w_norm = None

        self.create_genn_model(dt=dt)

    def create_genn_model(self, dt=1.0):
        self.genn_n_neurons = [(np.prod(self.tf_model.input_shape[1:]))]
        self.genn_w_name = []
        self.genn_w_inds = []
        self.genn_w_vals = []

        for layer in self.tf_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                self.genn_n_neurons.append(np.prod(layer.output_shape[1:]))
                tf_w = layer.get_weights()[0]
                genn_w = tf_w.copy()

                self.genn_w_name.append(layer.name)
                self.genn_w_inds.append(None)
                self.genn_w_vals.append(genn_w.flatten())

            elif isinstance(layer, tf.keras.layers.Conv2D):
                self.genn_n_neurons.append(np.prod(layer.output_shape[1:]))
                tf_w = layer.get_weights()[0]
                genn_w = np.full((self.genn_n_neurons[-2], self.genn_n_neurons[-1]), np.nan)

                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # No padding, such that output.shape < input.shape.
                if layer.padding == 'valid':
                    # oh = ceil((ih - kh + 1) / sh)
                    # ow = ceil((iw - kw + 1) / sw)

                    # For each post-synapse neuron (each column in genn_w):
                    for i_post in range(self.genn_n_neurons[-1]):

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
                    for i_post in range(self.genn_n_neurons[-1]):

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

                self.genn_w_name.append(layer.name)
                self.genn_w_inds.append(np.nonzero(~np.isnan(genn_w)))
                self.genn_w_vals.append(genn_w[self.genn_w_inds[-1]].flatten())


                #import matplotlib.pyplot as plt
                #plt.matshow(genn_w)
                #plt.matshow(genn_w[0::ic, 0::oc])


            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                self.genn_n_neurons.append(np.prod(layer.output_shape[1:]))
                genn_w = np.full((self.genn_n_neurons[-2], self.genn_n_neurons[-1]), np.nan)

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

                self.genn_w_name.append(layer.name)
                self.genn_w_inds.append(np.nonzero(~np.isnan(genn_w)))
                self.genn_w_vals.append(genn_w[self.genn_w_inds[-1]].flatten())

            elif isinstance(layer, tf.keras.layers.Flatten):
                # Ignore Flatten layers
                continue


        # PARAM VTHR (ORIGINAL)
        """
        # Define IF neuron class
        if_model = genn_model.create_custom_neuron_class(
            'if_model',
            param_names=['Vres', 'Vthr'],
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
            'Vthr': 1.0,
        }

        if_init = {
            'Vmem': 0.0,
            'Vmem_peak': 0.0,
            'nSpk': 0,
        }
        #"""

        # THREAD-LOCAL VTHR
        """
        # Define IF neuron class
        if_model = genn_model.create_custom_neuron_class(
            'if_model',
            param_names=['Vres'],
            var_name_types=[('Vmem', 'scalar'), ('Vmem_peak', 'scalar'), ('nSpk', 'unsigned int'), ('Vthr', 'scalar')],
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
            'Vthr': 1.0,
        }
        #"""

        # GLOBAL VTHR
        #"""
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
        #"""


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



        ########### POISSON INPUT ############
        '''
//----------------------------------------------------------------------------
// NeuronModels::Poisson
//----------------------------------------------------------------------------
//! Poisson neurons
/*! Poisson neurons have constant membrane potential (\c Vrest) unless they are
    activated randomly to the \c Vspike value if (t- \c SpikeTime ) > \c trefract.
                                                                                                                      
    It has 2 variables:
                                                                                                                      
    - \c V - Membrane potential (mV)
    - \c SpikeTime - Time at which the neuron spiked for the last time (ms)
                                                                                                                      
    and 4 parameters:
                                                                                                                      
    - \c trefract - Refractory period (ms)
    - \c tspike - duration of spike (ms)
    - \c Vspike - Membrane potential at spike (mV)
    - \c Vrest - Membrane potential at rest (mV)
                                                                                                                      
    \note The initial values array for the `Poisson` type needs two entries
    for `V`, and `SpikeTime` and the parameter array needs four entries for
    `therate`, `trefract`, `Vspike` and `Vrest`,  *in that order*.
                                                                                                                      
    \note This model uses a linear approximation for the probability
    of firing a spike in a given time step of size `DT`, i.e. the
    probability of firing is \f$\lambda\f$ times `DT`: \f$ p = \lambda \Delta t
    \f$. This approximation is usually very good, especially for typical,
    quite small time steps and moderate firing rates. However, it is worth
    noting that the approximation becomes poor for very high firing rates
    and large time steps.*/

class Poisson : public Base
{
public:
    DECLARE_MODEL(NeuronModels::Poisson, 4, 2);

    SET_SIM_CODE(
        "if(($(t) - $(spikeTime)) > $(tspike) && $(V) > $(Vrest)){\n"
        "   $(V) = $(Vrest);\n"
        "}"
        "else if(($(t) - $(spikeTime)) > $(trefract)){\n"
        "   if($(gennrand_uniform) < $(firingProb)[$(offset) + $(id)]){\n"
        "       $(V) = $(Vspike);\n"
        "       $(spikeTime) = $(t);\n"
        "   }\n"
        "}\n");
    SET_THRESHOLD_CONDITION_CODE("$(V) >= $(Vspike)");

    SET_PARAM_NAMES({"trefract", "tspike", "Vspike", "Vrest"});
    SET_VARS({{"V", "scalar"}, {"spikeTime", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"firingProb", "scalar*"}, {"offset", "unsigned int"}});
};
        '''



        # Define GeNN model
        self.genn_model = genn_model.GeNNModel('float', 'tg_model')
        self.genn_model.dT = dt

        self.neuron_pops = []
        self.synapse_pops = []



        # ========== INPUTS AS POISSON NEURONS
        # Add Poisson distributed spike inputs

        # ========== INPUTS WITH INJECTED CURRENT

        # Add input neurons
        self.neuron_pops.append(self.genn_model.add_neuron_population(
            'if0', self.genn_n_neurons[0], if_model, if_params, if_init)
        )

        self.neuron_pops[-1].set_extra_global_param('Vthr', 1.0)

        # Add current sources
        self.current_source = self.genn_model.add_current_source('cs', cs_model, 'if0', {}, cs_init)

        # For each synapse population
        for i, (w_inds, w_vals) in enumerate(zip(self.genn_w_inds, self.genn_w_vals)):

            # Add next layer of neurons
            self.neuron_pops.append(self.genn_model.add_neuron_population(
                'if' + str(i+1), self.genn_n_neurons[i+1], if_model, if_params, if_init)
            )

            self.neuron_pops[-1].set_extra_global_param('Vthr', 1.0)

            # Add synapses from last layer to this layer
            if w_inds is None: # Dense weight matrix
                self.synapse_pops.append(self.genn_model.add_synapse_population(
                    'syn' + str(i) + str(i+1), 'DENSE_INDIVIDUALG', genn_wrapper.NO_DELAY,
                    self.neuron_pops[-2], self.neuron_pops[-1],
                    'StaticPulse', {}, {'g': w_vals.copy()}, {}, {},
                    'DeltaCurr', {}, {})
                )

            else: # Sparse weight matrix
                self.synapse_pops.append(self.genn_model.add_synapse_population(
                    'syn' + str(i) + str(i+1), 'SPARSE_INDIVIDUALG', genn_wrapper.NO_DELAY,
                    self.neuron_pops[-2], self.neuron_pops[-1],
                    'StaticPulse', {}, {'g': w_vals.copy()}, {}, {},
                    'DeltaCurr', {}, {})
                )

                self.synapse_pops[-1].set_sparse_connections(w_inds[0].copy(), w_inds[1].copy())

        # Build and load model
        self.genn_model.build()
        self.genn_model.load()



        # for i in range(len(self.synapse_pops)):
        #     print('======= TG  ' + str(i))
        #     print(self.genn_w_vals[i].shape)
        #     if self.genn_w_inds[i] is not None:
        #         print(self.genn_w_inds[i][0].shape)
        #     print(self.genn_n_neurons[i])
        #     print(self.genn_n_neurons[i+1])
        # for i in range(len(self.synapse_pops)):
        #     print('===== GeNN  ' + str(i))
        #     self.genn_model.pull_var_from_device('syn' + str(i) + str(i+1), 'g')
        #     print(self.synapse_pops[i].vars['g'].view.shape)


        # import matplotlib.pyplot as plt
        # syn_pop = 0
        # plt.figure()
        # plt.plot(self.genn_w_vals[syn_pop])
        # plt.figure()
        # self.genn_model.pull_var_from_device('syn' + str(syn_pop) + str(syn_pop+1), 'g')
        # plt.plot(self.synapse_pops[syn_pop].vars['g'].view)
        # plt.show()
        # exit(1)




    def update_genn_weights(self, genn_w):
        # TODO ==========================================================
        pass


    def evaluate_genn_model(self, x, y, present_time=100.0, save_samples=[]):
        n_correct = 0
        n_examples = len(x)
        n_pops = len(self.neuron_pops)


        spike_idx = [[None for _ in enumerate(self.neuron_pops)] for _ in enumerate(save_samples)]     
        spike_times = [[None for _ in enumerate(self.neuron_pops)] for _ in enumerate(save_samples)]     


        # For each sample presentation
        for i in range(n_examples):
            # Before simulation
            for j, npop in enumerate(self.neuron_pops):
                npop.vars['Vmem'].view[:] = 0.0
                npop.vars['Vmem_peak'].view[:] = 0.0
                npop.vars['nSpk'].view[:] = 0
                self.genn_model.push_state_to_device('if' + str(j))


            # TODO: INPUT RATE ENCODING
            # FOR NOW, USE CONSTANT CURRENT INJECTION EQUAL TO INPUT MAGNITUDE
            self.current_source.vars['magnitude'].view[:] = x[i].flatten()
            self.genn_model.push_var_to_device('cs', 'magnitude')


            # Run simulation
            for t in range(math.ceil(present_time / self.genn_model.dT)):
                self.genn_model.step_time()

                # Save spikes
                if i in save_samples:
                    try:
                        k = save_samples.index(i)
                        for j, npop in enumerate(self.neuron_pops):
                            self.genn_model.pull_current_spikes_from_device(npop.name)

                            idx = npop.current_spikes    # size of npop
                            ts = np.ones(idx.shape) * t  # size of npop

                            if spike_idx[k][j] is None:
                                spike_idx[k][j] = np.copy(idx)
                                spike_times[k][j] = ts      
                            else:
                                spike_idx[k][j] = np.hstack((spike_idx[k][j], idx))
                                spike_times[k][j] = np.hstack((spike_times[k][j], ts))
                    except ValueError:
                        pass

            # After simulation
            self.genn_model.pull_var_from_device('if' + str(n_pops - 1), 'nSpk')
            nSpk_view = self.neuron_pops[n_pops - 1].vars['nSpk'].view
            n_correct += (np.argmax(nSpk_view) == y[i])

        accuracy = (n_correct / n_examples) * 100.

        return accuracy, spike_idx, spike_times
