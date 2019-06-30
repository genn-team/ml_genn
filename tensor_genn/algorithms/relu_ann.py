import numpy as np
import random
import math

import tensorflow as tf
from pygenn import genn_model, genn_wrapper

class ReLUANN():
    def __init__(self,neuron_resting_voltage=-60.0,neuron_threshold_voltage=-55.0,
        membrane_capacitance=1.0, model_timestep=1.0, single_example_time=350.):
        self.Vres = neuron_resting_voltage
        self.Vthr = neuron_threshold_voltage
        self.Cm = membrane_capacitance # For DNN - 1.0, CNN - 0.4
        self.timestep = model_timestep
        self.single_example_time = single_example_time

    def create_weight_matrices(self, tf_model):
        tf_weights = tf_model.get_weights()
        tf_layers = tf_model.layers

        g_model_weights = []
        n_units = [np.prod(tf_layers[0].input_shape[1:])]
        j=1
        for i, layer in enumerate(tf_layers):
            if not isinstance(layer,tf.keras.layers.Flatten):
                n_units.append(np.prod(layer.output_shape[1:]))
                syn_weights = np.zeros((n_units[j-1],n_units[j]))

                if isinstance(layer,tf.keras.layers.Conv2D):
                    kw,kh = layer.kernel_size
                    sw,sh = layer.strides
                    ih,iw,ic = layer.input_shape[1:]
                    oh,ow,oc = layer.output_shape[1:]
        
                    for n in range(int(n_units[j])):
                        for k in range(kh):
                            syn_weights[((n//oc)%ow)*ic*sw + ((n//oc)//ow)*ic*iw*sh + k*ic*iw:
                                        ((n//oc)%ow)*ic*sw + ((n//oc)//ow)*ic*iw*sh + k*ic*iw + kw*ic,
                                        n] = tf_weights[j-1][k,:,:,n%oc].reshape((-1))
                            
                elif isinstance(layer,tf.keras.layers.Dense):
                    syn_weights = tf_weights[j-1]
                g_model_weights.append(syn_weights)
                j += 1
        
        return g_model_weights, n_units

    def convert(self, tf_model):
        # Check model compatibility
        if not isinstance(tf_model,tf.keras.models.Sequential):
            raise NotImplementedError('Implementation for type {} models not found'.format(type(tf_model)))
        
        for layer in tf_model.layers[:-1]:
            if not isinstance(layer,(tf.keras.layers.Dense,tf.keras.layers.Flatten, tf.keras.layers.Conv2D)):
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

        # Fetch tf_model details
        n_layers = len(tf_model.layers)

        # Params and init
        if_params = {
            "Vres":self.Vres,
            "Vthr":self.Vthr,
            "Cm":self.Cm
        }
        if_init = {
            "Vmem":genn_model.init_var("Uniform", {"min": self.Vres, "max": self.Vthr}),
            "SpikeNumber":0
        }
        
        cs_init = {"magnitude":10.0}

        # Fetch augmented weight matrices
        g_weights, n_units = self.create_weight_matrices(tf_model)
        gw_inds = [np.nonzero(gw) for gw in g_weights]
        gw_vals = [g_weights[i][gw_inds[i]].reshape(-1) for i in range(len(g_weights))]

        # Define model and populations
        self.g_model = genn_model.GeNNModel("float","g_model")
        self.neuron_pops = []
        self.syn_pops = []

        for i in range(1,n_layers):
            if i == 1:
                # Presynaptic neuron
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    "if"+str(i-1),n_units[i-1],if_model,if_params,if_init)
                )
            
            # Postsynaptic neuron
            self.neuron_pops.append(self.g_model.add_neuron_population(
                "if"+str(i),n_units[i],if_model,if_params,if_init)
            )

            # Synapse
            self.syn_pops.append(self.g_model.add_synapse_population(
                "syn"+str(i-1)+str(i),"SPARSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
                self.neuron_pops[i-1],self.neuron_pops[i],
                "StaticPulse",{},{'g':gw_vals[i-1]},{},{},
                "DeltaCurr",{},{})
            )

            self.syn_pops[-1].set_sparse_connections(gw_inds[i-1][0],gw_inds[i-1][1])


        self.current_source = self.g_model.add_current_source("cs",cs_model,"if0",{},cs_init)

        self.g_model.dT = self.timestep
        self.g_model.build()
        self.g_model.load()

        return self.g_model, self.neuron_pops, self.current_source

    def evaluate(self, X, y=None):
        n_examples = len(X)
        X = X.reshape(n_examples,-1)
        y = y.reshape(n_examples)

        n = len(self.neuron_pops)
        
        n_correct = 0    

        for i in range(n_examples):
            # Before simulation
            for j, npop in enumerate(self.neuron_pops):
                    npop.vars["SpikeNumber"].view[:] = 0
                    npop.vars["Vmem"].view[:] = random.uniform(self.Vres,self.Vthr)
                    self.g_model.push_state_to_device("if"+str(j))
                
            self.current_source.vars['magnitude'].view[:] = X[i] / 100.
            self.g_model.push_var_to_device("cs",'magnitude')

            # Run simulation
            for _ in range(math.ceil(self.single_example_time/self.timestep)):
                self.g_model.step_time()

            # After simulation
            self.g_model.pull_var_from_device("if"+str(n-1),'SpikeNumber')
            SpikeNumber_view = self.neuron_pops[-1].vars["SpikeNumber"].view
            n_correct += (np.argmax(SpikeNumber_view)==y[i])

        accuracy = (n_correct / n_examples) * 100.
        
        return accuracy