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
    def __init__(self,neuron_resting_voltage=-60.0,neuron_threshold_voltage=-56.0,
        dense_membrane_capacitance=1.0,sparse_membrane_capacitance=0.2, 
        model_timestep=1.0, single_example_time=500.):
        self.Vres = neuron_resting_voltage
        self.Vthr = neuron_threshold_voltage
        self.dCm = dense_membrane_capacitance
        self.sCm = sparse_membrane_capacitance
        self.timestep = model_timestep
        self.single_example_time = single_example_time

    def create_weight_matrices(self, tf_model, scaled_tf_weights=None):
        if scaled_tf_weights is None:
            tf_weights = tf_model.get_weights()
        else:
            tf_weights = scaled_tf_weights
        tf_layers = tf_model.layers

        gw_inds = []
        gw_vals = []
        n_units = [np.prod(tf_layers[0].input_shape[1:])] # flatten layer shapes to (None, total_size)
        i = j = 1

        for layer in tf_layers:
            if not isinstance(layer, tf.keras.layers.Flatten):
                n_units.append(np.prod(layer.output_shape[1:]))
                syn_weights = np.zeros((n_units[j-1],n_units[j]))

                if isinstance(layer,tf.keras.layers.Dense):
                    syn_weights = tf_weights[i-1]
                    i += 1
                    gw_inds.append(None)
                    gw_vals.append(syn_weights.flatten())

                elif isinstance(layer,tf.keras.layers.Conv2D):
                    # Prepare weight matrices for Conv2D
                    '''
                    Indexing of neurons is done in the order: channels -> width -> height
                    Assign weights for every kw*ic block in the input and map to output neuron
                    corresponding to that convolution operation.
                    '''

                    kw,kh = layer.kernel_size
                    sw,sh = layer.strides 
                    ih,iw,ic = layer.input_shape[1:]
                    oh,ow,oc = layer.output_shape[1:]

                    if layer.padding == 'valid': # no padding in layer i.e. output_size < input_size
                        # Proceed with weight mapping for regular convolution
                        for n in range(int(n_units[j])): # output unit index/conv number
                            for k in range(kh): # over kernel height
                                syn_weights[((n//oc)%ow)*ic*sw + ((n//oc)//ow)*ic*iw*sh + k*ic*iw:
                                            ((n//oc)%ow)*ic*sw + ((n//oc)//ow)*ic*iw*sh + k*ic*iw + kw*ic,
                                            n] = tf_weights[i-1][k,:,:,n%oc].flatten()

                    elif layer.padding == 'same': # zero padding such that output_size == input_size
                        # Calculate padding for each side of input map (As done by tf.keras)
                        if (ih % sh == 0):
                            pad_along_height = max(kh - sh, 0)
                        else:
                            pad_along_height = max(kh - (ih % sh), 0)
                        if (iw % sw == 0):
                            pad_along_width = max(kw - sw, 0)
                        else:
                            pad_along_width = max(kw - (iw % sw), 0)
                        
                        pad_top = pad_along_height // 2 
                        pad_bottom = pad_along_height - pad_top
                        pad_left = pad_along_width // 2
                        pad_right = pad_along_width - pad_left

                        for n in range(int(n_units[j])):
                            # Calculate effective start and end indices on input matrix, along x and y dimensions
                            startw = max((n//oc)%ow - pad_left,0)*ic*sw
                            endw = min((n//oc)%ow + pad_right,iw-1)*ic*sw + ic
                            starth = max((n//oc)//ow - pad_top,0)*ic*iw*sh
                            endh = min((n//oc)//ow + pad_bottom,ih-1)*ic*iw*sh

                            # Calculate start and end indices for weight matrix to be assigned at the synapse
                            startkw = max(pad_left-((n*sw)//oc)%ow,0)
                            endkw = startkw + (endw-ic-startw)//ic + 1
                            startkh = max(pad_top-((n*sh)//oc)//ow,0)

                            # Weight mapping
                            for k in range(starth,endh+1,ic*iw):
                                syn_weights[k+startw:k+endw,n] = tf_weights[i-1][
                                    startkh + (k-starth)//(ic*iw),
                                    startkw:endkw,
                                    :,n%oc
                                ].flatten()
                    
                    i += 1
                    gw_inds.append(np.nonzero(syn_weights))
                    gw_vals.append(syn_weights[gw_inds[-1]].flatten())

                elif isinstance(layer,tf.keras.layers.AveragePooling2D):
                    pw, ph = layer.pool_size
                    sw, sh = layer.strides
                    ih, iw, ic = layer.input_shape[1:]
                    oh, ow, oc = layer.output_shape[1:]

                    for n in range(ow*oh): # output unit index
                        for k in range(ph): # over kernel height
                            for l in range(pw): # over kernel width
                                '''
                                Assign 1.0/(kernel size) as the weight from every input neuron to
                                corresponding output neuron
                                '''
                                syn_weights[(n%ow)*ic*sw + (n//ow)*ic*iw*sh + k*ic*iw + l*ic:
                                            (n%ow)*ic*sw + (n//ow)*ic*iw*sh + k*ic*iw + l*ic + ic,
                                            n*oc:n*oc+oc] = np.diag([1.0/(ph*pw)]*oc) # diag since we need a one-to-one mapping along channels
                    gw_inds.append(np.nonzero(syn_weights))
                    gw_vals.append(syn_weights[gw_inds[-1]].flatten())

                j += 1
        
        return gw_inds, gw_vals, n_units

    def convert(self, tf_model, scaled_tf_weights=None):
        supported_layers = (tf.keras.layers.Dense,tf.keras.layers.Flatten,tf.keras.layers.Conv2D,
                            tf.keras.layers.AveragePooling2D)

        # Check model compatibility
        if not isinstance(tf_model,tf.keras.models.Sequential):
            raise NotImplementedError('Implementation for type {} models not found'.format(type(tf_model)))
        
        for layer in tf_model.layers[:-1]:
            if not isinstance(layer,supported_layers):
                raise NotImplementedError('{} layers are not supported'.format(layer))
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.activation != tf.keras.activations.relu:
                    print(layer.activation)
                    raise NotImplementedError('Only ReLU activation function is supported')
                if layer.use_bias == True:
                    raise NotImplementedError('TensorFlow model should be trained without bias tensors')

        # create custom classes
        if_model = genn_model.create_custom_neuron_class(
            "if_model",
            param_names=["Vres","Vthr","Cm"],
            var_name_types=[("Vmem","scalar"),("SpikeNumber","unsigned int")],
            sim_code="""
            $(Vmem) += $(Isyn)*(DT / $(Cm));
            """,
            reset_code="""
            $(Vmem) = $(Vres); 
            $(SpikeNumber) += 1;
            """,
            threshold_condition_code="$(Vmem) >= $(Vthr)"
        )

        cs_model = genn_model.create_custom_current_source_class(
            "cs_model",
            var_name_types=[("magnitude","scalar")],
            injection_code="""
            $(injectCurrent, $(magnitude));
            """
        )

        # Params and init
        dense_if_params = {
            "Vres":self.Vres,
            "Vthr":self.Vthr,
            "Cm":self.dCm
        }

        sparse_if_params = {
            "Vres":self.Vres,
            "Vthr":self.Vthr,
            "Cm":self.sCm
        }

        if_init = {
            "Vmem":genn_model.init_var("Uniform", {"min": self.Vres, "max": self.Vthr}),
            "SpikeNumber":0
        }
        
        cs_init = {"magnitude":10.0}

        # Fetch augmented weight matrices
        gw_inds, gw_vals, n_units = self.create_weight_matrices(tf_model,scaled_tf_weights)

        # Define model and populations
        self.g_model = genn_model.GeNNModel("float","g_model")
        self.neuron_pops = []
        self.syn_pops = []

        for i,(inds,vals,n) in enumerate(zip(gw_inds,gw_vals,n_units),start=1):
            if i == 1:
                # Presynaptic neuron
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    "if"+str(i-1),n_units[i-1],if_model,sparse_if_params,if_init)
                )
            
            if inds is None:
                # Postsynaptic neuron
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    "if"+str(i),n_units[i],if_model,dense_if_params,if_init)
                )

                # Synapse
                self.syn_pops.append(self.g_model.add_synapse_population(
                    "syn"+str(i-1)+str(i),"DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
                    self.neuron_pops[-2],self.neuron_pops[-1],
                    "StaticPulse",{},{'g':vals},{},{},
                    "DeltaCurr",{},{})
                )
                
            else:
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    "if"+str(i),n_units[i],if_model,sparse_if_params,if_init)
                )

                self.syn_pops.append(self.g_model.add_synapse_population(
                    "syn"+str(i-1)+str(i),"SPARSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
                    self.neuron_pops[-2],self.neuron_pops[-1],
                    "StaticPulse",{},{'g':vals},{},{},
                    "DeltaCurr",{},{})
                )

                self.syn_pops[-1].set_sparse_connections(inds[0],inds[1])


        self.current_source = self.g_model.add_current_source("cs",cs_model,"if0",{},cs_init)

        self.g_model.dT = self.timestep
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
                npop.vars["Vmem"].view[:] = random.uniform(self.Vres,self.Vthr)
                self.g_model.push_state_to_device("if"+str(j))
                
            self.current_source.vars['magnitude'].view[:] = X[i]*(-self.Vthr * (self.dCm/self.timestep))
            self.g_model.push_var_to_device("cs",'magnitude')

            # Run simulation
            for t in range(math.ceil(self.single_example_time/self.timestep)):
                self.g_model.step_time()

                if i in save_example_spikes:
                    k = save_example_spikes.index(i)
                    for j,npop in enumerate(self.neuron_pops):
                        self.g_model.pull_current_spikes_from_device(npop.name)

                        ids = npop.current_spikes # size of npop
                        ts = np.ones(ids.shape) * t # size of npop
                    
                        if spike_ids[k][j] is None:
                            spike_ids[k][j] = np.copy(ids)
                            spike_times[k][j] = ts      
                        else:
                            spike_ids[k][j] = np.hstack((spike_ids[k][j],ids))
                            spike_times[k][j] = np.hstack((spike_times[k][j], ts))

            # After simulation
            self.g_model.pull_var_from_device("if"+str(n-1),'SpikeNumber')
            SpikeNumber_view = self.neuron_pops[-1].vars["SpikeNumber"].view
            n_correct += (np.argmax(SpikeNumber_view)==y[i])

        accuracy = (n_correct / n_examples) * 100.
        
        return accuracy, spike_ids, spike_times, self.neuron_pops, self.syn_pops