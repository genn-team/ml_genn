import numpy as np

from collections import defaultdict, namedtuple
from numbers import Number
from typing import Sequence, Union
from pygenn import GeNNModel
from pygenn.genn_wrapper import (SynapseMatrixConnectivity_PROCEDURAL,
                                 SynapseMatrixConnectivity_SPARSE,
                                 SynapseMatrixConnectivity_TOEPLITZ)
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from .compiled_network import CompiledNetwork
from ..initializers import Initializer
from ..network import Network

from copy import deepcopy
from pygenn.genn_model import (create_custom_custom_update_class,
                               create_custom_neuron_class,
                               create_custom_postsynaptic_class,
                               create_custom_weight_update_class, 
                               init_var)
from .weight_update_models import (static_pulse_model, static_pulse_delay_model,
                                   signed_static_pulse_model, 
                                   signed_static_pulse_delay_model)

WeightUpdateModel = namedtuple("WeightUpdateModel", ["model", "param_vals", "var_vals"])

def set_egps(var_egps, var_dict):
    for var, var_egp in var_egps.items():
        for p, value in var_egp.items():
            if isinstance(value, np.ndarray):
                var_dict[var].set_extra_global_init_param(p, value.flatten())
            else:
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
        param_name_types = []

    # If there aren't any variables already, add dictionary
    if not "var_name_types" in model_copy:
        model_copy["var_name_types"] = []

    # Convert any initializers to GeNN
    var_vals_copy = {}
    var_egp = {}
    for name, val in model.var_vals.items():
        if val.is_initializer:
            snippet = val.value.get_snippet()
            var_vals_copy[name] = init_var(snippet.snippet, 
                                           snippet.param_vals)
            var_egp[name] = snippet.egp_vals
        elif val.is_array:
            var_vals_copy[name] = val.value.flatten()
        else:
            var_vals_copy[name] = val.value
    
    # Loop through parameters in model
    model_copy["param_names"] = []
    constant_param_vals = {}
    for name, ptype in param_name_types:
        # Get value
        val = model.param_vals[name]
        
        # If value is a plain number, add it's name to parameter names
        if val.is_constant:
            model_copy["param_names"].append(name)
            constant_param_vals[name] = val.value
        # Otherwise, turn it into a (read-only) variable
        else:
            model_copy["var_name_types"].append((name, ptype, 
                                                 VarAccess_READ_ONLY))
            if val.is_initializer:
                snippet = val.value.get_snippet()
                var_vals_copy[name] = init_var(snippet.snippet,
                                               snippet.param_vals)
                var_egp[name] = snippet.egp_vals
            elif val.is_array:
                var_vals_copy[name] = val.value.flatten()
            else:
                var_vals_copy[name] = val

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
        self.prefer_in_memory_connect = prefer_in_memory_connect
        self.genn_kwargs = genn_kwargs

    def build_neuron_model(self, pop, model, custom_updates):
        # Build model customised for parameters and values
        model_copy, constant_param_vals, var_vals_copy, var_egp =\
            build_model(model)
        
        # Delete negative threshold condition if there is one
        # (this gets incorporated into weight update model)
        if "negative_threshold_condition_code" in model_copy:
            del model_copy["negative_threshold_condition_code"]

        # Return model and modified param and var values
        return model_copy, constant_param_vals, var_vals_copy, var_egp
    
    def build_synapse_model(self, conn, model, custom_updates):
        # Build model customised for parameters and values
        model_copy, constant_param_vals, var_vals_copy, var_egp =\
            build_model(model)

        # Return model and modified param and var values
        return model_copy, constant_param_vals, var_vals_copy, var_egp

    def build_weight_update_model(self, connection, connect_snippet, 
                                  custom_updates):
        # Build parameter values
        param_vals = {"g": connect_snippet.weight}
        het_delay = not connect_snippet.delay.is_constant
        if het_delay:
            param_vals["d"] = connect_snippet.delay
        
        # If source neuron model defines a negative threshold condition
        src_pop = connection.source()
        src_neuron_model = src_pop.neuron.get_model(src_pop, self.dt)
        if "negative_threshold_condition_code" in src_neuron_model:
            wum = WeightUpdateModel(
                (signed_static_pulse_delay_model if het_delay
                 else signed_static_pulse_model), param_vals, {})
            
            # Build model customised for parameters and values
            model_copy, constant_param_vals, var_vals_copy, var_egp =\
                build_model(wum)
            
            # Insert negative threshold condition code from neuron model
            model_copy["event_threshold_condition_code"] =\
                src_neuron_model["negative_threshold_condition_code"]
        else:
            wum = WeightUpdateModel(
                (static_pulse_delay_model if het_delay
                 else static_pulse_model), param_vals, {})
                 
            # Build model customised for parameters and values
            model_copy, constant_param_vals, var_vals_copy, var_egp =\
                build_model(wum)

        # Return model and modified param and var values
        return model_copy, constant_param_vals, var_vals_copy, {}, {}, var_egp
    
    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations):
        return CompiledNetwork(genn_model, neuron_populations, 
                               connection_populations)

    def compile(self, network: Network, name: str):
        genn_model = GeNNModel("float", name, **self.genn_kwargs)
        
        genn_model.dT = self.dt
        genn_model.batch_size = self.batch_size
        genn_model._model.set_seed(self.rng_seed)
        genn_model.timing_enabled = self.kernel_profiling
        
        # Loop through populations
        custom_updates = defaultdict(list)
        neuron_populations = {}
        for i, pop in enumerate(network.populations):
            # Check population has shape
            if pop.shape is None:
                raise RuntimeError("All populations need to have "
                                   "a shape before compiling network")
            
            # Build GeNN neuron model, parameters and values
            neuron = pop.neuron
            neuron_model, param_vals, var_vals, var_vals_egp =\
                self.build_neuron_model(pop, neuron.get_model(pop, self.dt), 
                                        custom_updates)
            
            # Create custom neuron model
            genn_neuron_model = create_custom_neuron_class("NeuronModel",
                                                           **neuron_model)
            # Add neuron population
            genn_pop = genn_model.add_neuron_population(
                f"Pop{i}", np.prod(pop.shape), 
                genn_neuron_model, param_vals, var_vals)
            
            # Configure EGPs
            set_egps(var_vals_egp, genn_pop.vars)
            
            # Add to neuron populations dictionary
            neuron_populations[pop] = genn_pop

        # Loop through connections
        connection_populations = {}
        for i, conn in enumerate(network.connections):
            # Build postsynaptic model
            syn = conn.synapse
            psm, psm_param_vals, psm_var_vals, psm_var_egp =\
                self.build_synapse_model(conn, syn.get_model(conn, self.dt),
                                         custom_updates)
            
            # Create custom postsynaptic model
            genn_psm = create_custom_postsynaptic_class("PostsynapticModel",
                                                        **psm)
            # Get connectivity init snippet
            connect_snippet =\
                conn.connectivity.get_snippet(conn, 
                                              self.prefer_in_memory_connect)
            # Build weight update model
            (wum, wum_param_vals, wum_var_vals, 
             wum_pre_var_vals, wum_post_var_vals, wum_var_egp) =\
                self.build_weight_update_model(conn, connect_snippet,
                                               custom_updates)
            
             # Create custom weight update model
            genn_wum = create_custom_weight_update_class("WeightUpdateModel",
                                                         **wum)
                                                     
            # If delays are constant, use as axonal delay otherwise, disable
            delay = conn.connectivity.delay
            axonal_delay = (delay.value if delay.is_constant
                            else 0)
            
            # Add synapse population
            genn_pop = genn_model.add_synapse_population(
                f"Syn{i}", connect_snippet.matrix_type, axonal_delay, 
                neuron_populations[conn.source()], neuron_populations[conn.target()],
                genn_wum, wum_param_vals, wum_var_vals, wum_pre_var_vals, wum_post_var_vals, 
                genn_psm, psm_param_vals, psm_var_vals,
                connectivity_initialiser=connect_snippet.snippet)
            
            # Configure EGPs
            set_egps(wum_var_egp, genn_pop.vars)
            set_egps(psm_var_egp, genn_pop.psm_vars)
            
            # Add to synapse populations dictionary
            connection_populations[conn] = genn_pop
        
        # Loop through custom updates added to model
        i = 0
        for cu_group, cu_list in custom_updates.items():
            for model, param_vals, var_vals, var_vals_egp, var_refs in cu_list:
                # Create customupdate model
                genn_cum = create_custom_custom_update_class("CustomUpdate",
                                                             **model)

                # Create variable references
                var_refs = {n: fn(neuron_populations, connection_populations)
                            for n, fn in var_refs.items()}
                # Add custom update
                genn_cu = genn_model.add_custom_update(f"CU{i}", cu_group, genn_cum,
                                                       param_vals, var_vals, var_refs)

                # Configure EGPs
                set_egps(var_vals_egp, genn_cu.vars)

                # Increment counter
                i+=1

        return self.create_compiled_network(genn_model, neuron_populations,
                                            connection_populations)
