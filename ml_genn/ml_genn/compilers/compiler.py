import inspect
import numpy as np
import os

from collections import defaultdict, namedtuple
from pygenn import CustomUpdateVarAccess, GeNNModel, VarAccess, VarAccessMode

from typing import List, Optional
from .compiled_network import CompiledNetwork
from .. import Connection, Population, Network
from ..callbacks import Callback
from ..communicators import Communicator
from ..utils.model import (CustomUpdateModel, NeuronModel,
                           SynapseModel, WeightUpdateModel)
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from copy import copy, deepcopy
from pygenn import (create_custom_update_model, create_neuron_model,
                    create_postsynaptic_model, create_weight_update_model,
                    create_var_ref, init_postsynaptic, init_weight_update)
from string import digits
from .weight_update_models import (static_pulse_model,
                                   static_pulse_delay_model,
                                   signed_static_pulse_model,
                                   signed_static_pulse_delay_model)
from ..utils.value import is_value_constant

# First pass of softmax - calculate max
softmax_1_model = {
    "var_name_types": [("MaxVal", "scalar", CustomUpdateVarAccess.REDUCE_NEURON_MAX)],
    "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY)],
    "update_code": """
    $(MaxVal) = $(Val);
    """}

# Second pass of softmax - calculate scaled sum of exp(value)
softmax_2_model = {
    "var_name_types": [("SumExpVal", "scalar", CustomUpdateVarAccess.REDUCE_NEURON_SUM)],
    "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY),
                 ("MaxVal", "scalar", VarAccessMode.READ_ONLY)],
    "update_code": """
    $(SumExpVal) = exp($(Val) - $(MaxVal));
    """}

# Third pass of softmax - calculate softmax value
softmax_3_model = {
    "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY),
                 ("MaxVal", "scalar", VarAccessMode.READ_ONLY),
                 ("SumExpVal", "scalar", VarAccessMode.READ_ONLY),
                 ("SoftmaxVal", "scalar")],
    "update_code": """
    $(SoftmaxVal) = exp($(Val) - $(MaxVal)) / $(SumExpVal);
    """}

def set_egp(egp_vals, egp_dict):
    for egp, value in egp_vals.items():
        if isinstance(value, np.ndarray):
            egp_dict[egp].set_values(value.flatten())
        else:
            egp_dict[egp].set_values(value)


def set_var_egps(var_egp_vals, var_dict):
    for var, var_egp in var_egp_vals.items():
        for p, value in var_egp.items():
            if isinstance(value, np.ndarray):
                var_dict[var].set_extra_global_init_param(p, value.flatten())
            else:
                var_dict[var].set_extra_global_init_param(p, value)

def create_reset_custom_update(reset_vars, var_ref_creator):
    # Create empty model
    model = CustomUpdateModel(model={"params": [],
                                     "var_refs": [],
                                     "update_code": ""},
                              param_vals={}, var_vals={}, var_refs={})

    # Loop through reset vars
    for name, type, value in reset_vars:
        # Add variable reference
        model.add_var_ref(name, type, var_ref_creator(name))

        # If variable should be reset to another variable
        if isinstance(value, str):
            # If variable to reset to isn't already in list
            # **TODO** give warning if it's not afterwards!
            existing_reset_var = [v for v in reset_vars if v[0] == value]
            if len(existing_reset_var) == 0:
                # Add read-only variable reference to other variable
                model.add_var_ref(value, type, var_ref_creator(value))
                model.set_var_ref_access_mode(value, VarAccessMode.READ_ONLY)

            # Add code to set var
            model.append_update_code(f"$({name}) = $({value});")
        # Otherwise
        else:
            # Add reset value parameter
            model.add_param(name + "Reset", type, value)

            # Add code to set var
            model.append_update_code(f"$({name}) = $({name}Reset);")

    return model

class SupportedMatrixType:
    def __init__(self, supported: List[int]):
        # Build dictionary of supported connectivity types and order
        self._supported = {v: i for i, v in enumerate(supported)}

    def get_best(self, available: List[int]) -> Optional[int]:
        # Intersect supported connectivity types with those that are available
        possible = self._supported.keys() & available

        # If there are no possible options
        if len(possible) == 0:
            return None
        # Otherwise, return the connectivity with 
        # the highest priority from possible
        else:
            return min(possible, key=lambda p: self._supported[p])

class ZeroInSyn(Callback):
    def __init__(self, genn_syn_pop):
        self.genn_syn_pop = genn_syn_pop

    def on_batch_begin(self, batch):
        self.genn_syn_pop.out_post.view[:]= 0.0
        self.genn_syn_pop.out_post.push_to_device()


class Compiler:
    def __init__(self, supported_matrix_type: List[int], dt: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 communicator: Communicator = None, **genn_kwargs):
        self.dt = dt
        self.full_batch_size = batch_size
        self.rng_seed = rng_seed
        self.kernel_profiling = kernel_profiling
        self.supported_matrix_type = SupportedMatrixType(supported_matrix_type)
        self.communicator = communicator
        self.genn_kwargs = genn_kwargs

        # If a communicator is provided
        if communicator is not None:
            # Check that batch size can be evenly divided between number of ranks
            if (batch_size % communicator.num_ranks) != 0:
                raise RuntimeError("Batch size must be divisible "
                                   "by total number of ranks")
            # Divide batch size by number of ranks
            self.batch_size = batch_size // communicator.num_ranks
        # Otherwise, batch size is un-modified
        else:
            self.batch_size = batch_size

    def pre_compile(self, network: Network, genn_model, **kwargs):
        return None

    def calculate_delay(self, conn: Connection, delay, compile_state):
        return delay

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state):
        model_copy = deepcopy(model)

        # Delete negative threshold condition if there is one
        # (this gets incorporated into weight update model)
        if "negative_threshold_condition_code" in model_copy.model:
            del model_copy.model["negative_threshold_condition_code"]

        return model_copy

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state):
        # Build model customised for parameters and values
        return model

    def build_weight_update_model(self, connection: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state):
        # Build parameter values
        param_vals = {"g": connect_snippet.weight}
        het_delay = not is_value_constant(connect_snippet.delay)
        if het_delay:
            param_vals["d"] = connect_snippet.delay

        # If source neuron model defines a negative threshold condition
        src_pop = connection.source()
        src_neuron_model = src_pop.neuron.get_model(src_pop, self.dt,
                                                    self.batch_size)
        if "negative_threshold_condition_code" in src_neuron_model.model:
            wum = WeightUpdateModel(
                (deepcopy(signed_static_pulse_delay_model) if het_delay
                 else deepcopy(signed_static_pulse_model)), param_vals)

            # Insert negative threshold condition code from neuron model
            wum.model["event_threshold_condition_code"] =\
                src_neuron_model.model["negative_threshold_condition_code"]

            return wum
        else:
            return WeightUpdateModel(
                (deepcopy(static_pulse_delay_model) if het_delay
                 else deepcopy(static_pulse_model)), param_vals)

    def add_custom_update(self, genn_model: GeNNModel,
                          model: CustomUpdateModel,
                          group: str, name: str):
        # Process model
        (cu_model, cu_param_vals, cu_var_vals,
         cu_egp_vals, cu_var_egp_vals, 
         cu_var_refs, cu_egp_refs) = model.process()

        # Create custom update model
        genn_cum = create_custom_custom_update_class("CustomUpdate",
                                                     **cu_model)

        # Add custom update
        genn_cu = genn_model.add_custom_update(name, group,
                                               genn_cum, cu_param_vals, 
                                               cu_var_vals, cu_var_refs,
                                               cu_egp_refs)

        # Configure var init EGPs
        set_var_egps(cu_var_egp_vals, genn_cu.vars)
        return genn_cu

    def add_softmax_custom_updates(self, genn_model, genn_pop, 
                                   input_var_name: str, output_var_name: str,
                                   custom_update_group_prefix: str = ""):
        # Create custom update model to implement 
        # first softmax pass and add to model
        softmax_1 = CustomUpdateModel(
            softmax_1_model, {}, {"MaxVal": 0.0},
            {"Val": create_var_ref(genn_pop, input_var_name)})

        genn_softmax_1 = self.add_custom_update(
            genn_model, softmax_1, 
            custom_update_group_prefix + "Softmax1",
            "CUSoftmax1" + genn_pop.name)

        # Create custom update model to implement 
        # second softmax pass and add to model
        softmax_2 = CustomUpdateModel(
            softmax_2_model, {}, {"SumExpVal": 0.0},
            {"Val": create_var_ref(genn_pop, input_var_name),
             "MaxVal": create_var_ref(genn_softmax_1, "MaxVal")})

        genn_softmax_2 = self.add_custom_update(
            genn_model, softmax_2, 
            custom_update_group_prefix + "Softmax2",
            "CUSoftmax2" + genn_pop.name)

        # Create custom update model to implement 
        # third softmax pass and add to model
        softmax_3 = CustomUpdateModel(
            softmax_3_model, {}, {},
            {"Val": create_var_ref(genn_pop, input_var_name),
             "MaxVal": create_var_ref(genn_softmax_1, "MaxVal"),
             "SumExpVal": create_var_ref(genn_softmax_2, "SumExpVal"),
             "SoftmaxVal": create_var_ref(genn_pop, output_var_name)})

        self.add_custom_update(
            genn_model, softmax_3, 
            custom_update_group_prefix + "Softmax3", 
            "CUSoftmax3" + genn_pop.name)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, compile_state):
        return CompiledNetwork(genn_model, neuron_populations,
                               connection_populations, self.communicator)

    def compile(self, network: Network, name: Optional[str] = None, **kwargs):
        # If no name is specifie
        if name is None:
            # Get the parent frame from our current frame
            # (whatever called compile)
            calframe = inspect.getouterframes(inspect.currentframe(), 1)

            # Extract name and path
            name = os.path.splitext(os.path.basename(calframe[1][1]))[0]

        # Strip out any non-alphanumerical characters from name
        clean_name = "".join(c for c in name if c.isalnum() or c == "_")
        clean_name = clean_name.lstrip(digits)

        # Add name of compiler type to name
        # **THINK** should this include compiler parameters?
        clean_name += type(self).__name__

        # If we are using a communicator, generate NCCL reductions
        # **NOTE** this only works with CUDA backend
        genn_kwargs = copy(self.genn_kwargs)
        if self.communicator is not None:
            genn_kwargs["enable_nccl_reductions"] = True

        # Create GeNN model and set basic properties
        genn_model = GeNNModel("float", clean_name, **genn_kwargs)
        genn_model.dt = self.dt
        genn_model.batch_size = self.batch_size
        genn_model.seed = self.rng_seed
        genn_model.timing_enabled = self.kernel_profiling

        # Run any pre-compilation logic
        compile_state = self.pre_compile(network, genn_model,
                                         **kwargs)

        # Loop through populations
        neuron_populations = {}
        for pop in network.populations:
            # Check population has shape
            if pop.shape is None:
                raise RuntimeError("All populations need to have "
                                   "a shape before compiling network")

            # Build GeNN neuron model, parameters and values
            neuron = pop.neuron
            neuron_model = neuron.get_model(pop, self.dt, self.batch_size)

            neuron_model, param_vals, var_vals, egp_vals, var_egp_vals =\
                self.build_neuron_model(
                    pop, neuron_model,
                    compile_state).process()

            # Create custom neuron model
            genn_neuron_model = create_neuron_model("NeuronModel",
                                                    **neuron_model)
            # Add neuron population
            genn_pop = genn_model.add_neuron_population(
                pop.name, np.prod(pop.shape),
                genn_neuron_model, param_vals, var_vals)

            # Configure spike and spike-like-event recording
            genn_pop.spike_recording_enabled = pop.record_spikes
            genn_pop.spike_event_recording_enabled = pop.record_spike_events

            # Configure EGPs
            set_egp(egp_vals, genn_pop.extra_global_params)

            # Configure var init EGPs
            set_var_egps(var_egp_vals, genn_pop.vars)

            # Add to neuron populations dictionary
            neuron_populations[pop] = genn_pop

        # Loop through connections
        connection_populations = {}
        for conn in network.connections:
            # Build postsynaptic model
            syn = conn.synapse
            (psm, psm_param_vals, psm_var_vals, 
             psm_egp_vals, psm_var_egp_vals) =\
                self.build_synapse_model(conn,
                                         syn.get_model(conn, self.dt,
                                                       self.batch_size),
                                         compile_state).process()

            # Create custom postsynaptic model
            genn_psm = create_postsynaptic_model("PostsynapticModel", **psm)
            # Get connectivity init snippet
            connect_snippet =\
                conn.connectivity.get_snippet(conn,
                                              self.supported_matrix_type)

            # Calculate delay
            delay = self.calculate_delay(conn, connect_snippet.delay,
                                         compile_state)

            # Build weight update model
            (wum, wum_param_vals, wum_var_vals,
             wum_egp_vals, wum_var_egp_vals,
             wum_pre_var_vals, wum_post_var_vals) =\
                self.build_weight_update_model(conn, connect_snippet,
                                               compile_state).process()

            # Create custom weight update model
            genn_wum = create_weight_update_model("WeightUpdateModel", **wum)

            # If delays are constant, use as axonal delay otherwise, disable
            axonal_delay = (delay if is_value_constant(delay)
                            else 0)

            # Add synapse population
            genn_pop = genn_model.add_synapse_population(
                conn.name, connect_snippet.matrix_type, axonal_delay,
                neuron_populations[conn.source()],
                neuron_populations[conn.target()],
                init_weight_update(genn_wum, wum_param_vals, wum_var_vals,
                                   wum_pre_var_vals, wum_post_var_vals),
                init_postsynaptic(genn_psm, psm_param_vals, psm_var_vals),
                connect_snippet.snippet)

            # If connectivity snippet has pre and postsynaptic
            # indices, set them in synapse group
            if (connect_snippet.pre_ind is not None 
                and connect_snippet.post_ind is not None):
                    genn_pop.set_sparse_connections(connect_snippet.pre_ind,
                                                    connect_snippet.post_ind)

            # Configure EGPs
            set_egp(wum_egp_vals, genn_pop.extra_global_params)
            set_egp(psm_egp_vals, genn_pop.psm_extra_global_params)

            # Configure var init EGPs
            set_var_egps(wum_var_egp_vals, genn_pop.vars)
            set_var_egps(psm_var_egp_vals, genn_pop.psm_vars)

            # Add to synapse populations dictionary
            connection_populations[conn] = genn_pop

        return self.create_compiled_network(genn_model, neuron_populations,
                                            connection_populations,
                                            compile_state)
