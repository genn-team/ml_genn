from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..neurons import LeakyIntegrateFire
from ..synapses import Delta
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType
from ..utils.model import CustomUpdateModel

from copy import deepcopy
from functools import partial
from pygenn.genn_model import create_var_ref, create_psm_var_ref
from .compiler import build_model
from ..utils.data import batch_dataset, get_metrics, get_dataset_size


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


class EPropCompiler(Compiler):
    def __init__(self, evaluate_timesteps: int, dt: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 prefer_in_memory_connect=True, 
                 reset_time_between_batches=True,
                 **genn_kwargs):
        super(EPropCompiler, self).__init__(dt, batch_size, rng_seed,
                                            kernel_profiling,
                                            prefer_in_memory_connect,
                                            **genn_kwargs)
        self.evaluate_timesteps = evaluate_timesteps
        self.reset_time_between_batches = reset_time_between_batches

    def build_neuron_model(self, pop, model, custom_updates,
                           pre_compile_output):
        # Make copy of model
        model_copy = deepcopy(model.model)

        # If population is an output
        if pop.neuron.output is not None:
            # if REGRESSION add "$(E) = $(Y) - $(YStar);"
            # else add whole CUDA horror after checking shape
        # Otherwise, if neuron isn't an input
        elif not hasattr(pop.neuron, set_input):
            # If model doesn't have variables or reset code, add empty
            # **YUCK**
            f "additional_input_vars" not in model_copy.model:
                model_copy.model["additional_input_vars"] = []
            if "var_name_types" not in model_copy.model:
                model_copy.model["var_name_types"] = []
            if "sim_code" not in model_copy.model:
                model_copy.model["sim_code"] = ""
            
            # Add additional input variable to receive feedback
            model_copy.model["additional_input_vars"] +=\
                [("ISynFeedback", "scalar", 0.0)],
            
            # Add additional state variable to store 
            # feedback and initialise to zero
            model_copy.model["var_name_types"] += ("E", "scalar")
            model_copy.var_vals["E"] = 0.0
            
            # Add sim code to store incoming feedback in new state variable
            model_copy.model["sim_code"] +=\
                f"\n$(E) = $(ISynFeedback);\n"
        
        # Build neuron model
        return super(EPropCompiler, self).build_neuron_model(
            pop, model, custom_updates, pre_compile_output)

    def build_synapse_model(self, conn, model, custom_updates,
                            pre_compile_output):        
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("EProp compiler only "
                                      "supports Delta synapses")


        return super(EPropCompiler, self).build_synapse_model(
            conn, model, custom_updates, pre_compile_output)
    
    def build_weight_update_model(self, conn, weight, delay,
                                  custom_updates, pre_compile_output):
        target_neuron = conn.target.neuron
        if isinstance(conn.target, LeakyIntegrateFire):
            if not target_neuron.integrate_during_refrac:
                raise NotImplementedError("")
            
            if not target.relative_reset:
                raise NotImplementedError("")
            
        return super(EPropCompiler, self).build_weight_update_model(
            conn, weight, delay, custom_updates, pre_compile_output)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, pre_compile_output):
        return CompiledInferenceNetwork(genn_model, neuron_populations,
                                        connection_populations,
                                        self.evaluate_timesteps,
                                        self.reset_time_between_batches)
