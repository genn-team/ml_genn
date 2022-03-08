import numpy as np

from numbers import Number
from typing import Sequence, Union
from pygenn import GeNNModel
from pygenn.genn_wrapper import (SynapseMatrixConnectivity_PROCEDURAL,
                                 SynapseMatrixConnectivity_SPARSE,
                                 SynapseMatrixConnectivity_TOEPLITZ)
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from ..initializers import Initializer
from ..model import Model

from copy import deepcopy
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_postsynaptic_class,
                               create_custom_weight_update_class, 
                               init_var)
from .weight_update_models import (static_pulse_model, static_pulse_delay_model,
                                   signed_static_pulse_model, 
                                   signed_static_pulse_delay_model)

def set_egps(var_egps, var_dict):
    for var, var_egp in var_egps.items():
        for p, value in var_egp.items():
            var_dict[var].set_extra_global_init_param(p, value)

def build_model(model):
    # Make copy of model
    model_copy = deepcopy(model.model)
    
    # Remove param names and types from copy of model (those that will "
    # be implemented as GeNN parameters will live in param_names)
    if "param_name_types" in model_copy:
        param_name_types = model_copy["param_name_types"]
        del model_copy["param_name_types"]
    else:
        param_name_types = {}
    
    # Convert any initializers to GeNN
    var_vals_copy = {}
    var_egp = {}
    for name, val in model.var_vals.items():
        if isinstance(val.value, Initializer):
            snippet = val.value.get_snippet()
            var_vals_copy[name] = init_var(snippet.snippet, 
                                           snippet.param_vals)
            var_egp[name] = snippet.egp_vals
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
    return model_copy, constant_param_vals, var_vals_copy, var_egp
    
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

    def build_neuron_model(self, model):
        # Build model customised for parameters and values
        model_copy, constant_param_vals, var_vals_copy, var_egp =\
            build_model(model)
        
        # Delete negative threshold condition if there is one
        # (this gets incorporated into weight update model)
        if "negative_threshold_condition_code" in model_copy:
            del model_copy["negative_threshold_condition_code"]

        # Create custom neuron model
        genn_neuron_model = create_custom_neuron_class("NeuronModel",
                                                       **model_copy)

        # Return model and modified param and var values
        return genn_neuron_model, constant_param_vals, var_vals_copy, var_egp
    
    def build_postsynaptic_model(self, model):
        # Build model customised for parameters and values
        model_copy, constant_param_vals, var_vals_copy, var_egp =\
            build_model(model)

        # Create custom postsynaptic model
        genn_psm = create_custom_postsynaptic_class("PostsynapticModel",
                                                    **model_copy)

        # Return model and modified param and var values
        return genn_psm, constant_param_vals, var_vals_copy, var_egp

    def build_weight_update_model(self, connection, connect_snippet):
        # Build parameter values
        param_vals = {"g": connect_snippet.weight}
        het_delay = not connect_snippet.delay.is_constant
        if het_delay:
            param_vals["d"] = connect_snippet.delay

        # If source neuron model defines a negative threshold condition
        src_pop = connection.source
        src_neuron_model = src_pop.neuron.get_model(src_pop)
        if "negative_threshold_condition_code" in src_neuron_model:
            # Build model customised for parameters and values
            model_copy, constant_param_vals, var_vals_copy, var_egp =\
                build_model((signed_static_pulse_delay_model if het_delay
                             else signed_static_pulse_model), param_vals, {})
            
            # Insert negative threshold condition code from neuron model
            model_copy["event_threshold_condition_code"] =\
                src_neuron_model["negative_threshold_condition_code"]
        else:
            # Build model customised for parameters and values
            model_copy, constant_param_vals, var_vals_copy, var_egp =\
                build_model((static_pulse_delay_model if het_delay 
                             else static_pulse_model), param_vals, {})
        
        # Create custom weight update model
        genn_wum = create_custom_weight_update_class("WeightUpdateModel",
                                                     **model_copy)
        # Return model and modified param and var values
        return genn_wum, constant_param_vals, var_vals_copy, {}, {}, var_egp

    def compile(self, model: Model, name: str):
        genn_model = GeNNModel("float", name, **self.genn_kwargs)
        
        genn_model.dT = self.dt
        genn_model.batch_size = self.batch_size
        genn_model._model.set_seed(self.rng_seed)
        genn_model.timing_enabled = self.kernel_profiling
        
        # Loop through populations
        neuron_populations = {}
        for i, pop in enumerate(model.populations):
            # Check population has shape
            if pop.shape is None:
                raise RuntimeError("All populations need to have "
                                   "a shape before compiling model")
            
            # Build GeNN neuron model, parameters and values
            neuron = pop.neuron
            neuron_model, param_vals, var_vals, var_vals_egp =\
                self.build_neuron_model(neuron.get_model(pop, self.dt))

            # Add neuron population
            genn_pop = genn_model.add_neuron_population(
                f"Pop{i}", np.prod(pop.shape), 
                neuron_model, param_vals, var_vals)
            
            # Configure EGPs
            set_egps(var_vals_egp, genn_pop.vars)
            
            # Add to neuron populations dictionary
            neuron_populations[pop] = genn_pop
            
        # Loop through inputs
        input_populations = {}
        for i, input in enumerate(model.inputs):
            # Check population has shape
            if input.shape is None:
                raise RuntimeError("All inputs need to have "
                                   "a shape before compiling model")
                                   
            # Build GeNN neuron model, parameters and values
            encoder = input.encoder
            neuron_model, param_vals, var_vals, var_vals_egp =\
                self.build_neuron_model(encoder.get_model(input, self.dt))
            
            # Add neuron population
            genn_pop = genn_model.add_neuron_population(
                f"Input{i}", np.prod(input.shape), 
                neuron_model, param_vals, var_vals)
            
            # Configure EGPs
            set_egps(var_vals_egp, genn_pop.vars)
            
            # Add to neuron populations dictionary
            neuron_populations[input] = genn_pop

        # Loop through connections
        connection_populations = {}
        for i, conn in enumerate(model.connections):
            # Build postsynaptic model
            syn = conn.synapse
            psm, psm_param_vals, psm_var_vals, psm_var_egp =\
                self.build_postsynaptic_model(syn.get_model(pop, self.dt))
            
            # Get connectivity init snippet
            connect_snippet =\
                conn.connectivity.get_snippet(conn, 
                                              self.prefer_in_memory_connect)

            # Build weight update model
            wum, wum_param_vals, wum_var_vals, wum_pre_var_vals, wum_post_var_vals, wum_var_egp =\
                self.build_weight_update_model(conn, connect_snippet)
        
            # If delays are constant, use as axonal delay otherwise, disable
            axonal_delay = (connect.delay.value if connect.delay.is_constant
                            else 0)
            
            # Add synapse population
            genn_pop = genn_model.add_synapse_population(
                self, f"Syn{i}", connect_snippet.matrix_type, axonal_delay, 
                neuron_populations[source], neuron_populations[target],
                wum, wum_param_vals, wum_var_vals, wum_pre_var_vals, wum_post_var_vals, 
                psm, psm_param_vals, psm_var_vals,
                connectivity_initialiser=connect_snippet.snippet)
            
            # Configure EGPs
            set_egps(wum_var_egp, genn_pop.vars)
            set_egps(psm_var_egp, genn_pop.psm_vars)
            
            # Add to synapse populations dictionary
            connection_populations[conn] = genn_pop
        
        return (genn_model, neuron_populations, input_populations, 
                connection_populations)
