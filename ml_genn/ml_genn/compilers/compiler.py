import inspect
import numpy as np
import os

from collections import defaultdict, namedtuple
from pygenn import (CustomUpdateVarAccess, GeNNModel,
                    VarAccess, VarAccessMode)
# **YUCK**
from pygenn.genn import SynapseGroup

from typing import List, Optional
from .compiled_network import CompiledNetwork
from .. import Connection, Population, Network
from ..callbacks import Callback
from ..communicators import Communicator
from ..utils.model import (CustomConnectivityUpdateModel, CustomUpdateModel,
                           NeuronModel, SynapseModel, WeightUpdateModel)
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from copy import copy, deepcopy
from pygenn import (create_custom_connectivity_update_model,
                    create_custom_update_model, create_den_delay_var_ref,
                    create_neuron_model, create_out_post_var_ref,
                    create_postsynaptic_model, create_weight_update_model,
                    create_var_ref, init_postsynaptic, init_weight_update)
from string import digits
from .weight_update_models import (get_static_pulse_delay_model, 
                                   get_signed_static_pulse_delay_model)
from ..utils.value import is_value_array, is_value_constant

from .weight_update_models import (static_pulse_model,
                                   signed_static_pulse_model)

# First pass of softmax - calculate max
softmax_1_model = {
    "vars": [("MaxVal", "scalar", CustomUpdateVarAccess.REDUCE_NEURON_MAX)],
    "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY)],
    "update_code": """
    MaxVal = Val;
    """}

# Second pass of softmax - calculate scaled sum of exp(value)
softmax_2_model = {
    "params": [("Temp", "scalar")],
    "vars": [("SumExpVal", "scalar", CustomUpdateVarAccess.REDUCE_NEURON_SUM)],
    "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY),
                 ("MaxVal", "scalar", VarAccessMode.READ_ONLY)],
    "update_code": """
    SumExpVal = exp((Val - MaxVal) / Temp);
    """}

# Third pass of softmax - calculate softmax value
softmax_3_model = {
    "params": [("Temp", "scalar")],
    "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY),
                 ("MaxVal", "scalar", VarAccessMode.READ_ONLY),
                 ("SumExpVal", "scalar", VarAccessMode.READ_ONLY),
                 ("SoftmaxVal", "scalar")],
    "update_code": """
    SoftmaxVal = exp((Val - MaxVal) / Temp) / SumExpVal;
    """}

def set_dynamic_param(param_names, set_param_dynamic):
    for p in param_names:
        set_param_dynamic(p, True)

def set_egp(egp_vals, egp_dict):
    for egp, value in egp_vals.items():
        if isinstance(value, np.ndarray):
            egp_dict[egp].set_init_values(value.flatten())
        else:
            egp_dict[egp].set_init_values(value)


def set_var_egps(var_egp_vals, var_dict):
    for var, var_egp in var_egp_vals.items():
        for p, value in var_egp.items():
            egp = var_dict[var].extra_global_params[p]
            if isinstance(value, np.ndarray):
                egp.set_init_values(value.flatten())
            else:
                egp.set_init_values(value)

def create_reset_custom_update(reset_vars, var_ref_creator):
    # Create empty model
    model = CustomUpdateModel(model={"params": [],
                                     "var_refs": [],
                                     "update_code": ""},
                              param_vals={}, var_vals={}, var_refs={})

    # Loop through reset vars
    broadcast_vars = set(r[0] for r in reset_vars)
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
            # Otherwise, remove it from set to broadcast
            else:
                broadcast_vars.remove(value)

            # Add code to set var
            model.append_update_code(f"{name} = {value};")
        # Otherwise
        else:
            # Add reset value parameter
            model.add_param(name + "Reset", type, value)

            # Add code to set var
            model.append_update_code(f"{name} = {name}Reset;")

    # Set broadcast access modes on variables for which this is possible
    for b in broadcast_vars:
        model.set_var_ref_access_mode(b, VarAccessMode.BROADCAST)

    return model

def create_local_var_refs(refs, genn_group):
    return {n: create_var_ref(genn_group, v) for n, v in refs.items()}

def get_delay_type(max_delay):
    if max_delay < 256:
       return "uint8_t"
    elif max_delay > 65535:
        raise NotImplementedError(f"Maximum delay steps exceeds 65535")
    else:
        return "uint16_t"

def get_conn_max_delay(conn, delay):
    # If maximum delay steps is specified
    if conn.max_delay_steps is not None:
        return 1 + conn.max_delay_steps
    # Otherwise, if delay is constant
    elif is_value_constant(delay):
        return 1 + round(delay)
    # Otherwise, if delays are specified as an array,
    # calculate maximum delay steps from array
    elif is_value_array(delay):
        return 1 + np.round(np.amax(delay)).astype(int)
    else:
        raise RuntimeError(f"Maximum delay associated with Connection "
                           f"{conn.name} cannot be determined "
                           f"automatically, please set max_delay_steps")

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


class Compiler:
    """Base class for all compilers"""
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
        """If any pre-processing is required before building neuron, synapse
        and weight update models, compilers should implement it here. Any 
        compiler-specific state that should be persistent across compilation
        should be encapsulated in an object returned from this method.
        
        Args:
            network:    Network to be compiled
            genn_model: Empty ``GeNNModel`` created at start of compilation
        """
        return None

    def apply_delay(self, genn_pop, conn: Connection,
                    delay, compile_state):
        """Apply delay to synapse population in compiler-specific manner
        
        Args:
            genn_pop:       GeNN synapse population to apply delay to
            conn:           Connection synapse model is associated with
            delay:          Base delay specified by connectivity
            compile_state:  Compiler-specific state created by
                            :meth:`.pre_compile`.
        """
        
        # If delays are constant, use as axonal delay
        if is_value_constant(delay):
            genn_pop.axonal_delay_steps = delay
        # Otherwise
        else:
            # Get maximum delay
            max_delay_steps = get_conn_max_delay(conn, delay)

            # Check delay fits in 16-bit limit
            if max_delay_steps > 65535:
                raise NotImplementedError(f"Maximum of {max_delay_steps}"
                                          f" delay steps for Connection "
                                          f"{conn.name} exceeds 65535")
            genn_pop.max_dendritic_delay_timesteps = max_delay_steps

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state) -> NeuronModel:
        """Apply compiler-specific processing to the base neuron model
        returned by :meth:`ml_genn.neurons.Neuron.get_model`.
        If modifications are made, this should be done to a (deep) copy.
        
        Args:
            pop:            Population neuron model is associated with
            model:          Base neuron model
            compile_state:  Compiler-specific state created by
                            :meth:`.pre_compile`.
        """
        model_copy = deepcopy(model)

        # Delete negative threshold condition if there is one
        # (this gets incorporated into weight update model)
        if "negative_threshold_condition_code" in model_copy.model:
            del model_copy.model["negative_threshold_condition_code"]

        return model_copy

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state) -> SynapseModel:
        """Apply compiler-specific processing to the base synapse model
        returned by :meth:`ml_genn.synapses.Synapse.get_model`.
        If modifications are made, this should be done to a (deep) copy.
        
        Args:
            conn:           Connection synapse model is associated with
            model:          Base synapse model
            compile_state:  Compiler-specific state created by
                            :meth:`.pre_compile`.
        """
        # Build model customised for parameters and values
        return model

    def build_weight_update_model(self, connection: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state) -> WeightUpdateModel:
        """Create compiler-specific weight update model for a connection.

        Args:
            connection:         Connection weight update 
                                model willl be used for
            connect_snippet:    Connectivity associated with connection
            compile_state:      Compiler-specific state created by
                                :meth:`.pre_compile`.
        """
        # Build parameter values
        param_vals = {"g": connect_snippet.weight}
        het_delay = not is_value_constant(connect_snippet.delay)
        if het_delay:
            # Get delay type to use for this connection
            delay_type = get_delay_type(
                get_conn_max_delay(connection, connect_snippet.delay))

            # If delays are specified as array, round
            # **NOTE** this is to prevent floating point delays e.g. those
            # obtained by Eventprop training being truncated later
            if is_value_array(connect_snippet.delay):
                param_vals["d"] = np.round(connect_snippet.delay)
            else:
                param_vals["d"] = connect_snippet.delay


        # If source neuron model defines a negative threshold condition
        src_pop = connection.source()
        src_neuron_model = src_pop.neuron.get_model(src_pop, self.dt,
                                                    self.batch_size)
        if "negative_threshold_condition_code" in src_neuron_model.model:
            wum = WeightUpdateModel(
                (get_signed_static_pulse_delay_model(delay_type) if het_delay
                 else deepcopy(signed_static_pulse_model)),
                param_vals)

            # Insert negative threshold condition code from neuron model
            wum.model["pre_event_threshold_condition_code"] =\
                src_neuron_model.model["negative_threshold_condition_code"]

            return wum
        else:
            return WeightUpdateModel(
                (get_static_pulse_delay_model(delay_type) if het_delay
                 else deepcopy(static_pulse_model)),
                param_vals)

    def add_custom_update(self, genn_model: GeNNModel,
                          model: CustomUpdateModel,
                          group: str, name: str):
        """Add a custom update to model.
        
        Args:
            genn_model: ``GeNNModel`` being compiled
            model:      Custom update model to add
            group:      Name of custom update group to associate update with
            name:       Name of custom update
        """
        # Process model
        (cu_model, cu_param_vals, cu_dynamic_param_names,
         cu_var_vals, cu_egp_vals, cu_var_egp_vals, 
         cu_var_refs, cu_egp_refs) = model.process()

        # Create custom update model
        genn_cum = create_custom_update_model("CustomUpdate",
                                              **cu_model)

        # Add custom update
        genn_cu = genn_model.add_custom_update(name, group,
                                               genn_cum, cu_param_vals, 
                                               cu_var_vals, cu_var_refs,
                                               cu_egp_refs)

        # Configure dynamic parameters
        set_dynamic_param(cu_dynamic_param_names, genn_cu.set_param_dynamic)

        # Configure var init EGPs
        set_var_egps(cu_var_egp_vals, genn_cu.vars)
        return genn_cu

    def add_custom_connectivity_update(self, genn_model: GeNNModel,
                                       model: CustomConnectivityUpdateModel,
                                       synapse_group: SynapseGroup, 
                                       group: str, name: str):
        (ccu_model, ccu_param_vals, ccu_dynamic_param_names,
         ccu_var_vals, ccu_egp_vals, ccu_var_egp_vals, ccu_pre_var_vals,
         ccu_post_var_vals, ccu_var_refs, ccu_pre_var_refs, ccu_post_var_refs, 
         ccu_egp_refs) = model.process()

        # Create custom conenctivity update model
        genn_ccum = create_custom_connectivity_update_model(
            "CustomConnectivityUpdate", **ccu_model)

        # Add custom connectivity update
        genn_ccu = genn_model.add_custom_connectivity_update(
            name, group, synapse_group, genn_ccum, 
            ccu_param_vals, ccu_var_vals, ccu_pre_var_vals, ccu_post_var_vals,
            ccu_var_refs, ccu_pre_var_refs, ccu_post_var_refs, ccu_egp_refs)

        # Configure dynamic parameters
        set_dynamic_param(ccu_dynamic_param_names, genn_ccu.set_param_dynamic)

        # Configure var init EGPs
        set_var_egps(ccu_var_egp_vals, genn_ccu.vars)
        return genn_ccu

    def add_out_post_zero_custom_update(self, genn_model, genn_syn_pop,
                                        group: str, name: str):
        # Build list of variables to reset
        has_delay = (genn_syn_pop.max_dendritic_delay_timesteps > 1)
        out_post_vars = [("OutPost", "scalar", 0.0)]
        if has_delay:
            out_post_vars.append(("DenDelay", "scalar", 0.0))

        # Create reset model
        zero_out_post_model = create_reset_custom_update(
            out_post_vars,
            lambda name: (create_out_post_var_ref(genn_syn_pop) 
                          if name == "OutPost"
                          else create_den_delay_var_ref(genn_syn_pop)))
        
        # Add GeNN custom update to model
        self.add_custom_update(
            genn_model, zero_out_post_model, 
            group, name)

    def add_softmax_custom_updates(self, genn_model, genn_pop, 
                                   input_var_name: str, output_var_name: str,
                                   custom_update_group_prefix: str = "",
                                   temperature: float = 1.0):
        """Adds a numerically stable softmax to the model:
        
        .. math::

            \\text{softmax}(x_i) = \\frac{e^{x_i - \\text{max}(x)}}{\\sum_j e^{x_j - \\text{max}(x)}}
        
        This softmax can then be calculated by triggering custom update groups
        "Softmax1", "Softmax2" and "Softmax3" in sequence (with optional prefix)
        
        Args:
            genn_model:                 ``GeNNModel`` being compiled
            genn_pop:                   GeNN population input and output 
                                        variables are associated with
            input_var_name:             Name of variable to read ``x`` from
            output_var_name:            Name of variable to write softmax to
            custom_update_group_prefix: Optional prefix to add to names of 
                                        custom update groups (enabling softmax
                                        operations required by different parts
                                        of the model to be triggered 
                                        seperately)
        """
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
            softmax_2_model, {"Temp": temperature}, {"SumExpVal": 0.0},
            {"Val": create_var_ref(genn_pop, input_var_name),
             "MaxVal": create_var_ref(genn_softmax_1, "MaxVal")})

        genn_softmax_2 = self.add_custom_update(
            genn_model, softmax_2, 
            custom_update_group_prefix + "Softmax2",
            "CUSoftmax2" + genn_pop.name)

        # Create custom update model to implement 
        # third softmax pass and add to model
        softmax_3 = CustomUpdateModel(
            softmax_3_model, {"Temp": temperature}, {},
            {"Val": create_var_ref(genn_pop, input_var_name),
             "MaxVal": create_var_ref(genn_softmax_1, "MaxVal"),
             "SumExpVal": create_var_ref(genn_softmax_2, "SumExpVal"),
             "SoftmaxVal": create_var_ref(genn_pop, output_var_name)})

        self.add_custom_update(
            genn_model, softmax_3, 
            custom_update_group_prefix + "Softmax3", 
            "CUSoftmax3" + genn_pop.name)

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict, compile_state):
        """Perform any final compiler-specific modifications to compiled 
        ``GeNNModel`` and return :class:`ml_genn.compilers.CompiledNetwork`
        derived object.
        
        Args:
            genn_model:             ``GeNNModel`` with all neuron and synapse
                                    groups added
            neuron_populations:     dictionary mapping 
                                    :class:`ml_genn.Population` objects
                                    to GeNN ``NeuronGroup`` objects they
                                    have been compiled into
            connection_populations: dictionary mapping 
                                    :class:`ml_genn.Connection` objects
                                    to GeNN ``SynapseGroup`` objects they
                                    have been compiled into
            compile_state:          Compiler-specific state created by
                                    :meth:`.pre_compile`.
            """
        return CompiledNetwork(genn_model, neuron_populations,
                               connection_populations, self.communicator)

    def compile(self, network: Network, name: Optional[str] = None, **kwargs):
        """Compiles network
        
        Args:
            network:    Network to compile
            name:       Optional name for model used to determine directory
                        to generate code to. If not specified, name of module
                        calling this function will be used.
            kwargs:     Keyword arguments passed to :meth:`.pre_compile`.

        Returns:
            Compiled network
        """
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

            # **HACK** only the first rank should build any code so turn of CUDA block size optimization
            if self.communicator.rank != 0:
                from pygenn.cuda_backend import BlockSizeSelect
                genn_kwargs["block_size_select_method"] = BlockSizeSelect.MANUAL

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

            (neuron_model, param_vals, dynamic_param_names, 
             var_vals, egp_vals, var_egp_vals) =\
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

            # Configure dynamic parameters
            set_dynamic_param(dynamic_param_names, genn_pop.set_param_dynamic)

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
            (psm, psm_param_vals, psm_dynamic_param_names, psm_var_vals,
             psm_egp_vals, psm_var_egp_vals, psm_neuron_var_refs) =\
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

            # Build weight update model
            (wum, wum_param_vals, wum_dynamic_param_names, wum_var_vals,
             wum_egp_vals, wum_var_egp_vals,
             wum_pre_var_vals, wum_post_var_vals,
             wum_pre_neuron_var_refs, wum_post_neuron_var_refs) =\
                self.build_weight_update_model(conn, connect_snippet,
                                               compile_state).process()

            # Create custom weight update model
            genn_wum = create_weight_update_model("WeightUpdateModel", **wum)

            # Get pre and postsynaptic GeNN populations
            pre_genn_group = neuron_populations[conn.source()]
            post_genn_group = neuron_populations[conn.target()]

            # Use to create local variable references
            psm_neuron_var_refs = create_local_var_refs(psm_neuron_var_refs,
                                                        post_genn_group)
            wum_pre_neuron_var_refs = create_local_var_refs(
                wum_pre_neuron_var_refs, pre_genn_group)
            wum_post_neuron_var_refs = create_local_var_refs(
                wum_post_neuron_var_refs, post_genn_group)
    
            # Add synapse population
            genn_pop = genn_model.add_synapse_population(
                conn.name, connect_snippet.matrix_type,
                pre_genn_group, post_genn_group,
                init_weight_update(genn_wum, wum_param_vals, wum_var_vals,
                                   wum_pre_var_vals, wum_post_var_vals,
                                   wum_pre_neuron_var_refs, 
                                   wum_post_neuron_var_refs),
                init_postsynaptic(genn_psm, psm_param_vals, psm_var_vals,
                                  psm_neuron_var_refs),
                connect_snippet.snippet)
            
            # Apply delay
            self.apply_delay(genn_pop, conn, connect_snippet.delay,
                             compile_state)

            # If connectivity snippet has pre and postsynaptic
            # indices, set them in synapse group
            if (connect_snippet.pre_ind is not None 
                and connect_snippet.post_ind is not None):
                    genn_pop.set_sparse_connections(connect_snippet.pre_ind,
                                                    connect_snippet.post_ind)

            # Configure dynamic parameters
            set_dynamic_param(wum_dynamic_param_names,
                              genn_pop.set_wu_param_dynamic)
            set_dynamic_param(psm_dynamic_param_names,
                              genn_pop.set_ps_param_dynamic)

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
