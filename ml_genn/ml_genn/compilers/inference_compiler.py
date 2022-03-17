from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler
from .compiled_model import CompiledModel
from ..utils import CustomUpdateModel

from functools import partial
from pygenn.genn_model import create_var_ref, create_psm_var_ref
from .compiler import build_model

#CompiledInferenceModel = type("CompiledInferenceModel", (CompiledModel, InferenceMixin), 
#    dict(CompiledModel.__dict__))

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
    
class InferenceCompiler(Compiler):
    def __init__(self, dt:float=1.0, batch_size:int=1, rng_seed:int=0,
                 kernel_profiling:bool=False, prefer_in_memory_connect=True,
                 **genn_kwargs):
        super(InferenceCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling, 
                                                prefer_in_memory_connect,
                                                **genn_kwargs)

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
