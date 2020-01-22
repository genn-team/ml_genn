import numpy as np
import random
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from pygenn import genn_model, genn_wrapper

'''
References: 
Peter U. Diehl, Daniel Neil, Jonathan Binas, Matthew Cook, Shih-Chii Liu, and Michael Pfeiffer. 2015. 
Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing. IJCNN (2015)
'''

class ReLUANN():
    def __init__(self, neuron_resting_voltage=-60.0, neuron_threshold_voltage=-56.0,
                 dense_membrane_capacitance=1.0, sparse_membrane_capacitance=0.2, 
                 dt=1.0, single_example_time=500.0):
        self.Vres = neuron_resting_voltage
        self.Vthr = neuron_threshold_voltage
        self.Cm_d = dense_membrane_capacitance
        self.Cm_s = sparse_membrane_capacitance
        self.dt = dt
        self.single_example_time = single_example_time

    def create_weight_matrices(self, tf_model, scaled_tf_wt=None):
        if scaled_tf_wt is None:
            tf_model.set_weights(scaled_tf_wt)

        genn_n_neurons = [(np.prod(tf_model.input_shape[1:]))]
        genn_weight_inds = []
        genn_weight_vals = []

        for layer in tf_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                genn_n_neurons.append(np.prod(layer.output_shape[1:]))
                tf_weights = layer.get_weights()[0]
                genn_weights = tf_weights

                genn_weight_inds.append(None)
                genn_weight_vals.append(genn_weights.flatten())

            elif isinstance(layer, tf.keras.layers.Conv2D):
                genn_n_neurons.append(np.prod(layer.output_shape[1:]))
                tf_weights = layer.get_weights()[0]
                genn_weights = np.full((genn_n_neurons[-2], genn_n_neurons[-1]), np.nan)

                kh, kw = layer.kernel_size
                sh, sw = layer.strides 
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                # No padding, such that output.shape < input.shape.
                if layer.padding == 'valid':
                    # oh = ceil((ih - kh + 1) / sh)
                    # ow = ceil((iw - kw + 1) / sw)

                    # For each post-synapse neuron (each column in genn_weights):
                    for i_post in range(genn_n_neurons[-1]):

                        # Find its output channel index and output pixel coordinates.
                        out_channel    = i_post % oc
                        out_pixel_flat = i_post // oc
                        out_pixel_row  = out_pixel_flat // ow
                        out_pixel_col  = out_pixel_flat % ow

                        # Find the row (pre-synapse neuron) offset of genn_weights corresponding to the input pixel
                        # at the first row and first column of the convolution kernel's current stride.
                        in_row_offset = out_pixel_row * sh * ic * iw    # input row offset
                        in_col_offset = out_pixel_col * sw * ic         # input col offset
                        pre_offset    = in_row_offset + in_col_offset

                        # For each row of the kernels that filter each input channel into this output channel,
                        # place that row (all input channels) into genn_weights at each kernel row block offset.
                        for k_row in range(kh):
                            k_row_start = pre_offset + k_row * ic * iw  # kernel row start in genn_weights
                            k_row_end   = k_row_start + ic * kw         # kernel row end in genn_weights
                            k_row_wt    = tf_weights[k_row, :, :, out_channel]
                            genn_weights[k_row_start : k_row_end, i_post] = k_row_wt.flatten()

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


                    # kh = 5, sh = 1
                    # in shape  = (12, 12, 16)
                    # out_shape = (12, 12, 8)

                    # pad_along_height = 0 if sh >= kh

                    # pad_along_height  =  kh - sh                     =  4
                    # pad_top           =  pad_along_height // 2       =  2
                    # pad_bottom        =  pad_along_height - pad_top  =  2



                    # For each post-synapse neuron (each column in genn_weights):
                    for i_post in range(genn_n_neurons[-1]):

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
                            wt = tf_weights[startkh + (k-starth)//(ic*iw), startkw : endkw, :, out_channel]
                            genn_weights[k+startw : k+endw, i_post] = wt.flatten()

                genn_weight_inds.append(np.nonzero(~np.isnan(genn_weights)))
                genn_weight_vals.append(genn_weights[genn_weight_inds[-1]].flatten())


                import matplotlib.pyplot as plt
                #plt.matshow(genn_weights)
                plt.matshow(genn_weights[0::ic, 0::oc])


            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                genn_n_neurons.append(np.prod(layer.output_shape[1:]))
                genn_weights = np.full((genn_n_neurons[-2], genn_n_neurons[-1]), np.nan)

                ph, pw = layer.pool_size
                sh, sw = layer.strides
                ih, iw, ic = layer.input_shape[1:]
                oh, ow, oc = layer.output_shape[1:]

                for n in range(ow * oh): # output unit index
                    for k in range(pw): # over kernel width
                        for l in range(ph): # over kernel height
                            # 1.0 / (kernel size) is the weight from input neurons to corresponding output neurons
                            genn_weights[(n % ow) * ic * sh + (n // ow) * ic * iw * sw + k * ic * iw + l * ic:
                                         (n % ow) * ic * sh + (n // ow) * ic * iw * sw + k * ic * iw + l * ic + ic,
                                         n * oc : n * oc + oc] = np.diag([1.0 / (pw * ph)] * oc) # diag since we need a one-to-one mapping along channels

                genn_weight_inds.append(np.nonzero(~np.isnan(genn_weights)))
                genn_weight_vals.append(genn_weights[genn_weight_inds[-1]].flatten())

            elif isinstance(layer, tf.keras.layers.Flatten):
                # Ignore Flatten layers
                continue

        return genn_weight_inds, genn_weight_vals, genn_n_neurons

    def convert(self, tf_model, scaled_tf_wt=None):
        supported_layers = (tf.keras.layers.Dense, tf.keras.layers.Flatten,
                            tf.keras.layers.Conv2D, tf.keras.layers.AveragePooling2D)

        # Check model compatibility
        if not isinstance(tf_model, tf.keras.Sequential):
            raise NotImplementedError('Implementation for type {} models not found'.format(type(tf_model)))
        for layer in tf_model.layers[:-1]:
            if not isinstance(layer, supported_layers):
                raise NotImplementedError('{} layers are not supported'.format(layer))
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.activation != tf.keras.activations.relu:
                    print(layer.activation)
                    raise NotImplementedError('Only ReLU activation function is supported')
                if layer.use_bias == True:
                    raise NotImplementedError('TensorFlow model should be trained without bias tensors')

        # Fetch weight matrices
        weight_inds, weight_vals, n_neurons = self.create_weight_matrices(tf_model, scaled_tf_wt)

        # create custom classes
        if_model = genn_model.create_custom_neuron_class(
            'if_model',
            param_names=['Vres', 'Vthr', 'Cm'],
            var_name_types=[('Vmem', 'scalar'), ('SpikeNumber', 'unsigned int')],
            sim_code='''
            $(Vmem) += ($(Isyn) / $(Cm)) * DT;
            ''',
            reset_code='''
            $(SpikeNumber) += 1;
            $(Vmem) = $(Vres); 
            ''',
            threshold_condition_code='$(Vmem) >= $(Vthr)'
        )

        cs_model = genn_model.create_custom_current_source_class(
            'cs_model',
            var_name_types=[('magnitude', 'scalar')],
            injection_code='''
            $(injectCurrent, $(magnitude));
            '''
        )

        # Params and init
        dense_if_params = {
            'Vres': self.Vres,
            'Vthr': self.Vthr,
            'Cm':   self.Cm_d,
        }

        sparse_if_params = {
            'Vres': self.Vres,
            'Vthr': self.Vthr,
            'Cm':   self.Cm_s,
        }

        if_init = {
            'SpikeNumber': 0,
            'Vmem':        genn_model.init_var('Uniform', {'min': self.Vres, 'max': self.Vthr}),
        }
        
        cs_init = {
            'magnitude': 10.0,
        }

        # Define model and populations
        self.g_model = genn_model.GeNNModel('float', 'g_model')
        self.g_model.dT = self.dt

        self.neuron_pops = []
        self.synapse_pops = []

        # Add input neurons
        self.neuron_pops.append(self.g_model.add_neuron_population(
            'if0', n_neurons[0], if_model, sparse_if_params, if_init)
        )

        # Add current sources
        self.current_source = self.g_model.add_current_source('cs', cs_model, 'if0', {}, cs_init)

        for i, (inds, vals) in enumerate(zip(weight_inds, weight_vals)):
            if inds is None:
                # Add next layer of neurons
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    'if' + str(i+1), n_neurons[i+1], if_model, dense_if_params, if_init)
                )

                # Add synapses from last layer to this layer
                self.synapse_pops.append(self.g_model.add_synapse_population(
                    'syn' + str(i) + str(i+1), 'DENSE_INDIVIDUALG', genn_wrapper.NO_DELAY,
                    self.neuron_pops[-2], self.neuron_pops[-1],
                    'StaticPulse', {}, {'g': vals}, {}, {},
                    'DeltaCurr', {}, {})
                )

            else:
                # Add next layer of neurons
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    'if' + str(i+1), n_neurons[i+1], if_model, sparse_if_params, if_init)
                )

                # Add synapses from last layer to this layer
                self.synapse_pops.append(self.g_model.add_synapse_population(
                    'syn' + str(i) + str(i+1), 'SPARSE_INDIVIDUALG', genn_wrapper.NO_DELAY,
                    self.neuron_pops[-2], self.neuron_pops[-1],
                    'StaticPulse', {}, {'g': vals}, {}, {},
                    'DeltaCurr', {}, {})
                )

                self.synapse_pops[-1].set_sparse_connections(inds[0], inds[1])

        self.g_model.build()
        self.g_model.load()

        return self.g_model, self.neuron_pops, self.current_source

    def evaluate(self, X, y=None, save_example_spikes=[]):
        n_examples = len(X)
        X = X.reshape(n_examples,-1)
        y = y.reshape(n_examples)
        n = len(self.neuron_pops)
        
        n_correct = 0
        spike_ids = [[None for _ in enumerate(self.neuron_pops)] for _ in enumerate(save_example_spikes)]     
        spike_times = [[None for _ in enumerate(self.neuron_pops)] for _ in enumerate(save_example_spikes)]     

        for i in range(n_examples):
            # Before simulation
            for j, npop in enumerate(self.neuron_pops):
                npop.vars["SpikeNumber"].view[:] = 0
                npop.vars["Vmem"].view[:] = random.uniform(self.Vres, self.Vthr)
                self.g_model.push_state_to_device("if" + str(j))
                
            self.current_source.vars['magnitude'].view[:] = X[i] * (-self.Vthr * (self.Cm_d / self.dt))
            self.g_model.push_var_to_device("cs",'magnitude')

            # Run simulation
            for t in range(math.ceil(self.single_example_time / self.dt)):
                self.g_model.step_time()

                if i in save_example_spikes:
                    try:
                        k = save_example_spikes.index(i)
                        for j,npop in enumerate(self.neuron_pops):
                            self.g_model.pull_current_spikes_from_device(npop.name)

                            ids = npop.current_spikes # size of npop
                            ts = np.ones(ids.shape) * t # size of npop
                        
                            if spike_ids[k][j] is None:
                                spike_ids[k][j] = np.copy(ids)
                                spike_times[k][j] = ts      
                            else:
                                spike_ids[k][j] = np.hstack((spike_ids[k][j], ids))
                                spike_times[k][j] = np.hstack((spike_times[k][j], ts))
                    except ValueError:
                        pass

            # After simulation
            self.g_model.pull_var_from_device("if"+str(n-1),'SpikeNumber')
            SpikeNumber_view = self.neuron_pops[-1].vars["SpikeNumber"].view
            n_correct += (np.argmax(SpikeNumber_view)==y[i])

        accuracy = (n_correct / n_examples) * 100.
        
        return accuracy, spike_ids, spike_times, self.neuron_pops, self.synapse_pops
