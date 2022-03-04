import numpy as np

from numbers import Number
from pygenn import GeNNModel
from pygenn.genn_model import create_custom_neuron_class, init_var
from ..initializers import Initializer
from ..model import Model

from copy import deepcopy

class Compiler:
    def __init__(self, dt:float=1.0, batch_size:int=1, rng_seed:int=0, kernel_profiling:bool=False, **genn_kwargs):
        self.dt = dt
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self.kernel_profiling = kernel_profiling
        self.genn_kwargs = genn_kwargs
    
    #def build_neuron_model(self, neuron_model: dict, param_vals: dict, var_vals: dict):
        # Make copy of neuron model
        neuron_model_copy = deepcopy(neuron_model)
        
        # Remove param names and types from copy of neuron model
        # (those that will be implemented as GeNN parameters will live in param_names)
        param_name_types = neuron_model_copy["param_name_types"]
        del neuron_model_copy["param_name_types"]
        
        # Convert any initializers to GeNN
        var_vals_copy = {}
        for name, val in var_vals.items():
            if isinstance(val.value, Initializer):
                var_vals_copy[name] = init_var(val.value.snippet, val.value.param_vals)
            else:
                var_vals_copy[name] = val.value
        
        # Loop through parameters in model
        neuron_model_copy["param_names"] = []
        constant_param_vals = {}
        for name, ptype in param_name_types:
            # Get value
            val = param_vals[name].value
            
            # If value isn't a plain number so can't be expressed 
            # as a GeNN parameter, turn it into a variable
            if not isinstance(val, Number):
                neuron_model_copy["var_name_types"].append((name, ptype))
                if isinstance(val, Initializer):
                    var_vals_copy[name] = init_var(val.snippet, val.param_vals)
                else:
                    var_vals_copy[name] = val
            # Otherwise, add it's name to parameter names
            else:
                neuron_model_copy["param_names"].append(name)
                constant_param_vals[name] = val
              
        # Create custom neuron model
        genn_neuron_model = create_custom_neuron_class("NeuronModel", 
                                                       **neuron_model_copy)
        
        # Return model and modified param and var values
        return genn_neuron_model, constant_param_vals, var_vals_copy
        
    def compile(self, model: Model, name):
        genn_model = GeNNModel("float", name, **self.genn_kwargs)
        
        genn_model.dT = self.dt
        genn_model.batch_size = self.batch_size
        genn_model._model.set_seed(self.rng_seed)
        genn_model.timing_enabled = self.kernel_profiling
        
        # Loop through populations
        for i, pop in enumerate(model.populations):
            # Build GeNN neuron model, parameters and values
            neuron = pop.neuron
            neuron_model, param_vals, var_vals = self.build_neuron_model(
                neuron.get_model(pop), neuron.param_vals, neuron.var_vals)
            
            genn_model.add_neuron_population(f"Population{i}", np.prod(pop.shape), 
                                             neuron_model, param_vals, var_vals)
