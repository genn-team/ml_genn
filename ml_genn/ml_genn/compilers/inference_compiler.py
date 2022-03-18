import numpy as np

from typing import Sequence, Union
from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..utils import CustomUpdateModel

from functools import partial
from pygenn.genn_model import create_var_ref, create_psm_var_ref
from .compiler import build_model
from ..utils.data import batch_numpy, get_numpy_size

def _build_reset_model(model, custom_updates, var_ref_creator):
    # If model has any state variables
    reset_vars = []
    if "var_name_types" in model.model:
        # Loop through them
        for v in model.model["var_name_types"]:
            # If variable either has default (read-write) 
            # access or this is explicitly set
            # **TODO** mechanism to exclude variables from reset
            if len(v) < 3 or (v[2] & VarAccessMode_READ_WRITE) != 0:
                reset_vars.append((v[0], v[1], model.var_vals[v[0]]))
 
    # If there's nothing to reset, return
    if len(reset_vars) == 0:
        return

    # Create empty model
    model = CustomUpdateModel(model={"param_name_types": [],
                                     "var_refs": [],
                                     "update_code": ""},
                              param_vals={}, var_vals={}, var_refs={})

    # Loop through reset vars
    for name, type, value in reset_vars:
        # Add variable reference and reset value parameter
        model.model["var_refs"].append((name, type))
        model.model["param_name_types"].append((name + "Reset", type))

        # Add parameter value and function to create variable reference
        # **NOTE** we use partial rather than a lambda so name is captured
        model.param_vals[name + "Reset"] = value
        model.var_refs[name] = partial(var_ref_creator, name=name)

        # Add code to set var 
        model.model["update_code"] += f"$({name}) = $({name}Reset);\n"
    
    # Build custom update model customised for parameters and values
    custom_model = build_model(model)
    
    # Add to list of custom updates to be applied in "Reset" group
    custom_updates["Reset"].append(custom_model + (model.var_refs,))

class CompiledInferenceNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations, 
                 connection_populations, evaluate_timesteps:int):
        super(CompiledInferenceNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations)
    
        self.evaluate_timesteps = evaluate_timesteps

    def evaluate_numpy(self, x: dict, y: dict):
        """ Evaluate an input in numpy format against labels
        accuracy --  dictionary containing accuracy of predictions 
                     made by each output Population or Layer
        """
        # Determine the number of elements in x and y
        x_size = get_numpy_size(x)
        y_size = get_numpy_size(x)
        
        if x_size is None:
            raise RuntimeError("Each input population must be "
                               " provided with same number of inputs")
        if y_size is None:
            raise RuntimeError("Each output population must be "
                               " provided with same number of labels")
        if x_size != y_size:
            raise RuntimeError("Number of inputs and labels must match")
        
        total_correct = {p: 0 for p in y.keys()}
        
        # Batch x and y
        batch_size = self.genn_model.batch_size
        x = batch_numpy(x, batch_size, x_size)
        y = batch_numpy(y, batch_size, y_size)
        
        # Loop through batches
        for x_batch, y_batch in zip(x, y):
            # Get predictions from batch
            correct = self.evaluate_batch(x_batch, y_batch)
            
            # Add to total correct
            for p, c in correct.items():
                total_correct[p] += c
        
        # Return dictionary containing correct count
        return {p: c / x_size for p, c in total_correct.items()}
    
    def evaluate_batch(self, x: dict, y: dict):
        """ Evaluate a single batch of inputs against labels
        Args:
        x --        dict mapping input Population or InputLayer to 
                    array containing one batch of inputs
        y --        dict mapping output Population or Layer to
                    array containing one batch of labels

        Returns:
        correct --  dictionary containing number of correct predictions 
                    made by each output Population or Layer
        """
        self.custom_update("Reset")

        # Apply inputs to model
        self.set_input(x)
        
        for t in range(self.evaluate_timesteps):
            self.step_time()

        # Get predictions from model
        y_star = self.get_output(list(y.keys()))
        
        # Return dictionaries of output population to number of correct
        # **TODO** insert loss-function/other metric here
        return {p : np.sum((np.argmax(o_y_star[:len(o_y)], axis=1) == o_y))
                for (p, o_y), o_y_star in zip(y.items(), y_star)}

class InferenceCompiler(Compiler):
    def __init__(self, evaluate_timesteps:int, dt:float=1.0, batch_size:int=1, rng_seed:int=0,
                 kernel_profiling:bool=False, prefer_in_memory_connect=True,
                 **genn_kwargs):
        super(InferenceCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling, 
                                                prefer_in_memory_connect,
                                                **genn_kwargs)
        self.evaluate_timesteps = evaluate_timesteps

    def build_neuron_model(self, pop, model, custom_updates):
        _build_reset_model(model, custom_updates, 
            lambda nrn_pops, _, name: create_var_ref(nrn_pops[pop], name))
    
        # Build neuron model
        return super(InferenceCompiler, self).build_neuron_model(
            pop, model, custom_updates)

    def build_synapse_model(self, conn, model, custom_updates):
        _build_reset_model(model, custom_updates, 
            lambda _, conn_pops, name: create_psm_var_ref(conn_pops[conn], name))
    
        return super(InferenceCompiler, self).build_synapse_model(
            conn, model, custom_updates)
    
    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations):
        return CompiledInferenceNetwork(genn_model, neuron_populations, 
                                        connection_populations,
                                        self.evaluate_timesteps)
