import numpy as np
import random

import tensorflow as tf
from pygenn import genn_model, genn_wrapper

class ReLUANN():
    def __init__(self,neuron_resting_voltage=-60.0,neuron_threshold_voltage=-55.0,
        membrane_capacitance=1.0, model_timestep=1.0, single_example_time=350.):
        self.Vres = neuron_resting_voltage
        self.Vthr = neuron_threshold_voltage
        self.Cm = membrane_capacitance
        self.timestep = model_timestep
        self.single_example_time = single_example_time

    def convert(self, tf_model):
        # Check model compatibility
        if not isinstance(tf_model,tf.keras.models.Sequential):
            raise NotImplementedError('Implementation for type {} models not found'.format(type(tf_model)))
        
        for layer in tf_model.layers[:-1]:
            if not isinstance(layer,(tf.keras.layers.Dense,tf.keras.layers.Flatten)):
                raise NotImplementedError('Only Dense and Flatten layers are supported')
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
        tf_weights = tf_model.get_weights()

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

        # Define model and populations
        self.g_model = genn_model.GeNNModel("float","g_model")
        self.neuron_pops = []

        for i in range(1,n_layers): # 1,2 - synapses
            if i == 1:
                # Presynaptic neuron
                self.neuron_pops.append(self.g_model.add_neuron_population(
                    "if"+str(i-1),tf_weights[i-1].shape[0],if_model,if_params,if_init)
                )

            # Postsynaptic neuron
            self.neuron_pops.append(self.g_model.add_neuron_population(
                "if"+str(i),tf_weights[i-1].shape[1],if_model,if_params,if_init)
            )

            # Synapse
            self.g_model.add_synapse_population(
                "syn"+str(i-1)+str(i),"DENSE_INDIVIDUALG",genn_wrapper.NO_DELAY,
                self.neuron_pops[i-1], self.neuron_pops[i],
                "StaticPulse",{},{'g':tf_weights[i-1].reshape(-1)},{},{},
                "DeltaCurr",{},{}
            )
        
        self.current_source = self.g_model.add_current_source("cs",cs_model,"if0",{},cs_init)

        self.g_model.dT = self.timestep
        self.g_model.build()
        self.g_model.load()

        return self.g_model

    def evaluate(self, X, y=None):
        n_examples = len(X)
        X = X.reshape(n_examples,-1)
        y = y.reshape(n_examples)

        runtime = n_examples * self.single_example_time
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
            while self.g_model.t < self.single_example_time * (i+1):
                self.g_model.step_time()

            # After simulation
            self.g_model.pull_var_from_device("if"+str(n-1),'SpikeNumber')
            SpikeNumber_view = self.neuron_pops[-1].vars["SpikeNumber"].view
            n_correct += (np.argmax(SpikeNumber_view)==y[i])

        accuracy = (n_correct / n_examples) * 100.
        
        return accuracy