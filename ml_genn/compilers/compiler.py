import numpy as np

from numbers import Number
from typing import Sequence, Union
from pygenn import GeNNModel
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from ..initializers import Initializer
from ..model import Model

from copy import deepcopy
from pygenn.genn_model import (create_custom_neuron_class, 
                               create_custom_weight_update_class, 
                               init_var)
from .weight_update_models import (static_pulse_model, static_pulse_delay_model,
                                   signed_static_pulse_model, 
                                   signed_static_pulse_delay_model)

class Compiler:
    def __init__(self, dt:float=1.0, batch_size:int=1, rng_seed:int=0,
                 kernel_profiling:bool=False, prefer_in_memory_connect=True,
                 **genn_kwargs):
        self.dt = dt
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self.kernel_profiling = kernel_profiling
        self.prefer_in_memory_connect = True
        self.genn_kwargs = genn_kwargs
    
    def build_model(model):
        # Make copy of model
        model_copy = deepcopy(model.model)
        
        # Remove param names and types from copy of model (those that will "
        # be implemented as GeNN parameters will live in param_names)
        param_name_types = model_copy["param_name_types"]
        del model_copy["param_name_types"]
        
        # Convert any initializers to GeNN
        var_vals_copy = {}
        for name, val in model.var_vals.items():
            if isinstance(val.value, Initializer):
                var_vals_copy[name] = init_var(val.value.snippet, 
                                               val.value.param_vals)
            else:
                var_vals_copy[name] = val.value
        
        # Loop through parameters in model
        model_copy["param_names"] = []
        constant_param_vals = {}
        for name, ptype in param_name_types:
            # Get value
            val = model.param_vals[name].value
            
            # If value isn't a plain number so can't be expressed 
            # as a GeNN parameter, turn it into a (read-only) variable
            if not isinstance(val, Number):
                model_copy["var_name_types"].append((name, ptype, 
                                                     VarAccess_READ_ONLY))
                if isinstance(val, Initializer):
                    var_vals_copy[name] = init_var(val.snippet,
                                                   val.param_vals)
                else:
                    var_vals_copy[name] = val
            # Otherwise, add it's name to parameter names
            else:
                model_copy["param_names"].append(name)
                constant_param_vals[name] = val
        
        # Return modified model and; params and var values
        return model_copy, constant_param_vals, var_vals_copy

    def build_neuron_model(self, model):
        # Build model customised for parameters and values
        model_copy, constant_param_vals, var_vals_copy =\
            build_model(model)
        
        # Delete negative threshold condition if there is one
        # (this gets incorporated into weight update model)
        del model_copy["negative_threshold_condition_code"]

        # Create custom neuron model
        genn_neuron_model = create_custom_neuron_class("NeuronModel",
                                                       **model_copy)

        # Return model and modified param and var values
        return genn_neuron_model, constant_param_vals, var_vals_copy
    
    def build_postsynaptic_model(self, model):
        # Build model customised for parameters and values
        model_copy, constant_param_vals, var_vals_copy =\
            build_model(model)

        # Create custom postsynaptic model
        genn_psm = create_custom_postsynaptic_class("PostsynapticModel",
                                                    **model_copy)

        # Return model and modified param and var values
        return genn_psm, constant_param_vals, var_vals_copy
    """
    def build_weight_update_model(self, connection, weights, delays):
        # Build parameter values
        param_vals = {"g": weights}
        het_delay = not delays.is_constant
        if het_delay:
            param_vals["d"] = delays

        # If source neuron model defines a negative threshold condition
        src_pop = connection.source
        src_neuron_model = src_pop.neuron.get_model(src_pop)
        if "negative_threshold_condition_code" in src_neuron_model:
            # Build model customised for parameters and values
            model_copy, constant_param_vals, var_vals_copy = build_model(
                signed_static_pulse_delay_model if het_delay else signed_static_pulse_model, 
                param_vals, {})
            
            # Insert negative threshold condition code from neuron model
            model_copy["event_threshold_condition_code"] =\
                src_neuron_model["negative_threshold_condition_code"]
        else:
            # Build model customised for parameters and values
            model_copy, constant_param_vals, var_vals_copy = build_model(
                static_pulse_delay_model if het_delay else static_pulse_model, 
                param_vals, {})
        
        # Create custom weight update model
        genn_wum = create_custom_weight_update_class("WeightUpdateModel",
                                                     **model_copy)
        # Return model and modified param and var values
        return genn_wum, constant_param_vals, var_vals_copy
    """
    def compile(self, model: Model, name):
        genn_model = GeNNModel("float", name, **self.genn_kwargs)
        
        genn_model.dT = self.dt
        genn_model.batch_size = self.batch_size
        genn_model._model.set_seed(self.rng_seed)
        genn_model.timing_enabled = self.kernel_profiling
        
        # Loop through populations
        neuron_populations = {}
        for i, pop in enumerate(model.populations):
            # Build GeNN neuron model, parameters and values
            neuron = pop.neuron
            neuron_model, param_vals, var_vals =\
                self.build_neuron_model(neuron.get_model(pop, self.dt))
            
            # Add neuron population
            genn_pop = genn_model.add_neuron_population(
                f"Pop{i}", np.prod(pop.shape), 
                neuron_model, param_vals, var_vals)

            # Add to neuron populations dictionary
            neuron_populations[pop] = genn_pop

        # Loop through connections
        synapse_populations = {}
        for i, conn in enumerate(model.connections):
            # Build postsynaptic model
            syn = conn.synapse
            connect = conn.connectivity
            psm, psm_param_vals, psm_var_vals =\
                self.build_postsynaptic_model(syn.get_model(pop, self.dt))
            
            # Get connectivity init snippet
            connect_snippet =\
                connect.get_snippet(self.prefer_in_memory_connect)

            # Build weight update model
            #wum, wum_param_vals, wum_var_vals = self.build_weight_update_model(
            #    conn, connect.weight, connect.delay, connect_snippet.matrix_mode)
            
            # If delays are constant, use as axonal delay otherwise, disable
            axonal_delay = (connect.delay.value if connect.delay.is_constant
                            else 0)
            
            genn_pop = genn_model.add_synapse_population(
                self, f"Syn{i}", matrix_type, axonal_delay, 
                neuron_populations[source], neuron_populations[target]
                wum, wum_param_vals, wum_var_vals, wum_pre_var_vals, wum_post_var_vals, 
                psm, psm_param_vals, psm_var_vals,
                connectivity_initialiser=connect_snippet.conn_init):
            
            # Add to synapse populations dictionary
            synapse_populations[conn] = genn_pop