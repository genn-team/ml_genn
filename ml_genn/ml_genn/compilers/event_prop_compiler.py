import logging
import numpy as np
import sympy

from string import Template
from typing import Iterator, Sequence, Union
from pygenn import (CustomUpdateVarAccess, VarAccess, VarAccessMode,
                    SynapseMatrixType, SynapseMatrixWeight)

from .compiler import Compiler
from .compiled_training_network import CompiledTrainingNetwork
from .. import Connection, Population, Network
from ..callbacks import (BatchProgressBar, Callback, CustomUpdateOnBatchBegin,
                         CustomUpdateOnBatchEnd, CustomUpdateOnEpochEnd,
                         CustomUpdateOnTimestepEnd)
from ..communicators import Communicator
from ..connection import Connection
from ..losses import Loss, SparseCategoricalCrossentropy, MeanSquareError
from ..metrics import MetricsType
from ..neurons import Input
from ..optimisers import Optimiser
from ..readouts import (AvgVar, AvgVarExpWeight, FirstSpikeTime,
                        MaxVar, SumVar, Var)
from ..utils.auto_model import AutoModel, AutoNeuronModel, AutoSynapseModel
from ..utils.callback_list import CallbackList
from ..utils.model import (CustomUpdateModel, Model, NeuronModel, 
                           SynapseModel, WeightUpdateModel)
from ..utils.network import PopulationType
from ..utils.snippet import ConnectivitySnippet

from copy import deepcopy
from itertools import chain
from pygenn import (create_egp_ref, create_psm_var_ref,
                    create_var_ref, create_wu_var_ref)
from .compiler import create_reset_custom_update, get_delay_type
from ..utils.auto_tools import solve_ode
from ..utils.module import get_object, get_object_mapping
from ..utils.network import get_underlying_conn, get_underlying_pop
from ..utils.value import is_value_array, is_value_constant

from .compiler import softmax_1_model, softmax_2_model
from ..optimisers import default_optimisers
from ..losses import default_losses

logger = logging.getLogger(__name__)

# EventProp uses a fixed size ring-buffer structure with a read pointer to
# read data for the backward pass and a write pointer to write data from the
# forward pass. These start positioned like this:
# 
# RingReadEndOffset   RingWriteStartOffset
#        V                    V
#        |--------------------|--------------------
#        | Backward pass data | Forward pass data  
#        |--------------------|--------------------
#                            ^ ^
#            <- RingReadOffset RingWriteOffset ->
#
# and move in opposite direction until the end of the example:
#
# RingReadEndOffset   RingWriteStartOffset
#        V                    V
#        |--------------------|--------------------
#        | Backward pass data | Forward pass data |
#        |--------------------|--------------------
#        ^                                         ^
# <- RingReadOffset                          RingWriteOffset ->
#
# 
# Then, a reset custom update points the read offset  
# to the end of the previous forward pass's data and provides a new 
# space to write data from the next forward pass
#
#                     RingReadEndOffset    RingWriteStartOffset
#                             V                    V
#                             |--------------------|--------------------
#                             | Backward pass data | Forward pass data |
#                             |--------------------|--------------------
#                                                 ^ ^
#                                 <- RingReadOffset RingWriteOffset ->
#
# Because the buffer is circular and all incrementing and decrementing of 
# read and write offsets check for wraparound, this can continue forever
# **NOTE** due to the inprecision of ASCII diagramming there are out-by-one errors in the above

# Basic weight-update model for connections without delay
weight_update_model = {
    "params": [("weight", "scalar")], 
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "pre_spike_syn_code": """
    addToPost(weight);
    """,
    "pre_event_threshold_condition_code": """
    BackSpike_pre
    """}

gradient_batch_reduce_model = {
    "vars": [("ReducedGradient", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
    "var_refs": [("Gradient", "scalar")],
    "update_code": """
    ReducedGradient = Gradient;
    Gradient = 0;
    """}

spike_count_batch_reduce_model = {
    "var_refs": [("SpikeCount", "int"),
                 ("SpikeCountBatch", "int", VarAccessMode.REDUCE_SUM)],
    "update_code": """
    SpikeCountBatch = SpikeCount;
    SpikeCount = 0;
    """}

# Template used to generate backward passes for neurons
neuron_backward_pass = Template(
    """
    const int ringOffset = (batch * num_neurons * $max_spikes) + (id * $max_spikes);
    const scalar backT = $example_time - t - dt;

    // Backward pass
    $dynamics
    if (BackSpike) {
        $transition

        // Decrease read pointer
        RingReadOffset--;
        if (RingReadOffset < 0) {
            RingReadOffset = $max_spikes - 1;
        }
        BackSpike = false;
    }
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    if (RingReadOffset != RingReadEndOffset && (backT - RingSpikeTime[ringOffset + RingReadOffset] - dt) <= 0.0) {
        BackSpike = true;
    }

    // Forward pass
    """)


# Template used to generate reset code for neurons
neuron_reset = Template(
    """
    if(RingWriteOffset != RingReadEndOffset) {
        // Write spike time and I-V to tape
        RingSpikeTime[ringOffset + RingWriteOffset] = t;
        $write
        RingWriteOffset++;

        // Loop around if we've reached end of circular buffer
        if (RingWriteOffset >= $max_spikes) {
            RingWriteOffset = 0;
        }
    }
    $strict_check
    """)

# Code used to add optional strict checking after neuron reset
neuron_reset_strict_check = """
    else {
        printf("%f: hidden: ring buffer violation in neuron %d, read end offset: %d, write start offset: %d, read offset: %d, write offset: %d\\n", t, id, RingReadEndOffset, RingWriteStartOffset, RingReadOffset, RingWriteOffset);
        assert(false);
    }
    """

# Build basic weight-update model for connections with delay
def _get_delay_weight_update_model(delay_type):
    return {"params": [("delay", delay_type), ("weight", "scalar")],
            "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
                             
            "pre_spike_syn_code": """
            addToPostDelay(g, d);
            """,
            "pre_event_threshold_condition_code": """
            BackSpike_pre
            """}

def _get_conn_max_delay(conn, delay):
    # If maximum delay steps is specified
    if conn.max_delay_steps is not None:
        return conn.max_delay_steps
    # Otherwise, if delay is constant
    elif is_value_constant(delay):
        return 1 + delay
    # Otherwise, if delays are specified as an array,
    # calculate maximum delay steps from array
    elif is_value_array(delay):
        return np.amax(delay) + 1
    else:
        raise RuntimeError(f"Maximum delay associated with Connection "
                          f"{conn.name} cannot be determined "
                          f"automatically, please set max_delay_steps")

def _get_lmd_name(symbol: Union[str, sympy.Symbol]):
    symbol_name = (symbol.name if isinstance(symbol, sympy.Symbol)
                   else symbol)
    return f"Lambda{symbol_name}"

def _get_lmd_sym(symbol: Union[str, sympy.Symbol]):
    return sympy.Symbol(_get_lmd_name(symbol))

# one could reduce saved vars by solving the threshold equation for one of the vars and substituting the equation
def _simplify_using_threshold(var_names, thresold_expr, expr):
    # If no threshold expression is provided, return expression un-modified
    if thresold_expr == 0:
        return expr

    # Find first variable referenced by threshold condition
    try:
        the_var = next(sympy.Symbol(v) for v in var_names 
                       if thresold_expr.has(sympy.Symbol(v)))
    # If no variables are referenced, return expression un-modified
    except StopIteration:
        return expr

    # Solve threshold expression wrt variable
    sln = sympy.solve(thresold_expr, the_var)

    # If there is no solution, return un-modified expression
    if len(sln) != 1:
        return expr

    # Substitute variable from original expression 
    # with solution from threshold condition
    return expr.subs(the_var, sln[0])

def _add_required_parameters(model: AutoModel, genn_model: Model, expression):
    # If any target population parameters are 
    # referenced, duplicate in synapse model
    for n, v in model.param_vals.items():
        if (expression.has(sympy.Symbol(n)) and not genn_model.has_param(n)):
            genn_model.add_param(n, "scalar", v)

class CompileState:
    def __init__(self, losses, readouts, backend_name):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self.backend_name = backend_name
        self._optimiser_connections = []
        self._neuron_reset_vars = []
        self._synapse_reset_vars = []
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []
        self.spike_count_populations = []
        self.batch_softmax_populations = []
        self.timestep_softmax_populations = []
        self.feedback_connections = []
        self.update_trial_pops = []

    def add_optimiser_connection(self, conn, weight: bool, delay: bool):
        self._optimiser_connections.append((conn, weight, delay))

    def add_neuron_reset_vars(self, pop, reset_vars, 
                              reset_event_ring, reset_v_ring):
        self._neuron_reset_vars.append((pop, reset_vars, 
                                        reset_event_ring, reset_v_ring))
    
    def add_synapse_reset_vars(self, conn, reset_vars):
        self._synapse_reset_vars.append((conn, reset_vars))
        
    def create_reset_custom_updates(self, compiler, genn_model,
                                    neuron_pops, conn_pops):
        # Loop through neuron variables to reset
        for i, (pop, vars, r_ev, r_v) in enumerate(self._neuron_reset_vars):
            # Create reset model
            model = create_reset_custom_update(
                vars,
                lambda name: create_var_ref(neuron_pops[pop], name))

            # If we want to add code to reset event ring buffer
            if r_ev:
                # Add references to ring buffer offsets
                model.add_var_ref("RingReadOffset", "int",
                                  create_var_ref(neuron_pops[pop],
                                                 "RingReadOffset"))
                model.add_var_ref("RingWriteOffset", "int", 
                                  create_var_ref(neuron_pops[pop],
                                                 "RingWriteOffset"))
                model.add_var_ref("RingWriteStartOffset", "int", 
                                  create_var_ref(neuron_pops[pop],
                                                 "RingWriteStartOffset"))
                model.add_var_ref("RingReadEndOffset", "int", 
                                  create_var_ref(neuron_pops[pop],
                                                 "RingReadEndOffset"))

                # Add additional update code to update ring buffer offsets
                model.append_update_code(
                    f"""
                    RingReadOffset = RingWriteOffset - 1;
                    if (RingReadOffset < 0) {{
                        RingReadOffset = {compiler.max_spikes - 1};
                    }}
                    RingReadEndOffset = RingWriteStartOffset;
                    RingWriteStartOffset = RingReadOffset;
                    """)
            # Otherwise, if we want to add code to reset a voltage ring buffer
            elif r_v:
                # Add references to ring buffer offsets
                model.add_var_ref("RingReadOffset", "int",
                                  create_var_ref(neuron_pops[pop],
                                                 "RingReadOffset"))
                model.add_var_ref("RingWriteOffset", "int", 
                                  create_var_ref(neuron_pops[pop],
                                                 "RingWriteOffset"))
                # Add additional update code to update ring buffer offsets
                model.append_update_code(
                    f"""
                    RingReadOffset = RingWriteOffset;
                    if (RingWriteOffset >= {2 * compiler.example_timesteps}) {{
                        RingWriteOffset = 0;
                    }}
                    """)
            
            # Add custom update
            compiler.add_custom_update(genn_model, model, 
                                       "Reset", f"CUResetNeuron{i}")

        # Loop through synapse variables to reset
        for i, (conn, vars) in enumerate(self._synapse_reset_vars):
            # Create reset model
            model = create_reset_custom_update(
                vars,
                lambda name: create_psm_var_ref(conn_pops[conn], name))

            # Add custom update
            compiler.add_custom_update(genn_model, model, 
                                       "Reset", f"CUResetSynapse{i}")
    
    @property
    def optimiser_connections(self):
        return self._optimiser_connections

    @property
    def is_reset_custom_update_required(self):
        return (len(self._neuron_reset_vars) > 0
                or len(self._synapse_reset_vars) > 0)


class UpdateTrial(Callback):
    def __init__(self, genn_pop):
        self.genn_pop = genn_pop

    def on_batch_begin(self, batch: int):
        logger.debug(f"Updating trial at start of batch {batch}")

        # Set dynamic parameter to batch ID
        self.genn_pop.set_dynamic_param_value("Trial", batch)


class CustomUpdateOnLastTimestep(Callback):
    """Callback that triggers a GeNN custom update 
    at the start of the last timestep in each example"""
    def __init__(self, name: str, example_timesteps: int):
        self.name = name
        self.example_timesteps = example_timesteps
    
    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def on_timestep_begin(self, timestep: int):
        if timestep == (self.example_timesteps - 1):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of timestep {timestep}")
            self._compiled_network.genn_model.custom_update(self.name)


class CustomUpdateOnBatchEndNotFirst(Callback):
    """Callback that triggers a GeNN custom update 
    at the end of every batch after the first."""
    def __init__(self, name: str):
        self.name = name

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network
        
    def on_batch_end(self, batch, metrics):
        if batch > 0:
            logger.debug(f"Running custom update {self.name} "
                         f"at end of batch {batch}")
            self._compiled_network.genn_model.custom_update(self.name)

class CustomUpdateOnFirstBatchEnd(Callback):
    """Callback that triggers a GeNN custom update 
    at the end of first batch."""
    def __init__(self, name: str):
        self.name = name

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network
        
    def on_batch_end(self, batch, metrics):
        if batch == 0:
            logger.debug(f"Running custom update {self.name} "
                         f"at end of batch {batch}")
            self._compiled_network.genn_model.custom_update(self.name)

class EventPropCompiler(Compiler):
    """Compiler for training models using EventProp [Wunderlich2021]_.

    The EventProp compiler supports :class:`ml_genn.neurons.LeakyIntegrateFire`
    hidden neuron models; and :class:`ml_genn.losses.SparseCategoricalCrossentropy` loss functions for classification
    and :class:`ml_genn.losses.MeanSquareError` for regression.

    EventProp implements a fully event-driven backward pass meaning that its memory 
    overhead scales with the number of spikes per-trial rather than sequence length. 
    
    In the original paper, [Wunderlich2021]_ 
    derived EventProp to support loss functions of the form:

    .. math::

        {\\cal L} = l_p(t^{\\text{post}}) + \\int_0^T l_V(V(t),t) dt
    
    such as

    .. math::

        l_V= -\\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\log \\left( \\frac{\\exp\\left(V_{l(m)}^m(t)\\right)}{\\sum_{k=1}^{N_{\\text{class}}} \\exp\\left(V_{k}^m(t) \\right)} \\right)
    
    where a function of output neuron membrane voltage is calculated each
    timestep -- in mlGeNN, we refer to these as *per-timestep loss functions*.
    However, [Nowotny2024]_ showed that tasks with more complex temporal 
    structure cannot be learned using these loss functions and extended 
    the framework to support loss functions of the form:

    .. math::

        {\\cal L}_F = F\\left(\\textstyle \\int_0^T l_V(V(t),t) \\, dt\\right)
    
    such as:
    
    .. math::

        {\\mathcal L_{\\text{sum}}} = - \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\log \\left( \\frac{\\exp\\left(\\int_0^T V_{l(m)}^m(t) dt\\right)}{\\sum_{k=1}^{N_{\\text{out}}} \\exp\\left(\\int_0^T V_{k}^m(t) dt\\right)} \\right)

    where a function of the integral of voltage is calculated once per-trial.

    Args:
        example_timesteps:          How many timestamps each example will be
                                    presented to the network for
        losses:                     Either a dictionary mapping loss functions
                                    to output populations or a single loss
                                    function to apply to all outputs
        optimiser:                  Optimiser to use when applying weights
        reg_lambda_upper:           Regularisation strength, should typically
                                    be the same as ``reg_lambda_lower``.
        reg_lambda_lower:           Regularisation strength, should typically
                                    be the same as ``reg_lambda_upper``.
        reg_nu_upper:               Target number of hidden neuron
                                    spikes used for regularisation
        max_spikes:                 What is the maximum number of spikes each
                                    neuron (input and hidden) can emit each
                                    trial? This is used to allocate memory 
                                    for the backward pass.
        strict_buffer_checking:     For performance reasons, if neurons emit
                                    more than ``max_spikes`` they are normally
                                    ignored but, if this flag is set, 
                                    this will cause an error.
        per_timestep_loss:          Should we use the per-timestep or
                                    per-trial loss functions described above?
        dt:                         Simulation timestep [ms]
        ttfs_alpha                  TODO
        softmax_temperature         TODO
        batch_size:                 What batch size should be used for
                                    training? In our experience, EventProp works
                                    best with modest batch sizes (32-128)
        rng_seed:                   What value should GeNN's GPU RNG be seeded
                                    with? This is used for all GPU randomness
                                    e.g. weight initialisation and Poisson 
                                    spike train generation
        kernel_profiling:           Should GeNN record the time spent in each
                                    GPU kernel? These values can be extracted
                                    directly from the GeNN model which can be 
                                    accessed via the ``genn_model`` property
                                    of the compiled model.
        reset_time_between_batches: Should time be reset to zero at the start
                                    of each example or allowed to run
                                    continously? 
        communicator:               Communicator used for inter-process
                                    communications when training across
                                    multiple GPUs.
        delay_optimiser:            Optimiser to use when applying delays. If 
                                    None, ``optimiser`` will be used for delays
        delay_learn_conns:          Connection for which delays should be 
                                    learned as well as weight
        
    """

    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 reg_lambda_upper: float = 0.0, reg_lambda_lower: float = 0.0,
                 reg_nu_upper: float = 0.0, max_spikes: int = 500,
                 strict_buffer_checking: bool = False,
                 per_timestep_loss: bool = False, dt: float = 1.0,
                 ttfs_alpha: float = 0.01, softmax_temperature: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 solver: str = "exponential_euler",
                 communicator: Communicator = None,
                 delay_optimiser=None,
                 delay_learn_conns: Sequence = [],
                 **genn_kwargs):
        supported_matrix_types = [SynapseMatrixType.TOEPLITZ,
                                  SynapseMatrixType.PROCEDURAL_KERNELG,
                                  SynapseMatrixType.DENSE,
                                  SynapseMatrixType.SPARSE]
        super(EventPropCompiler, self).__init__(supported_matrix_types, dt,
                                                batch_size, rng_seed,
                                                kernel_profiling, solver,
                                                communicator,
                                                **genn_kwargs)
        self.example_timesteps = example_timesteps
        self.losses = losses
        self.reg_lambda_upper = reg_lambda_upper
        self.reg_lambda_lower = reg_lambda_lower
        self.reg_nu_upper = reg_nu_upper
        self.max_spikes = max_spikes
        self.strict_buffer_checking = strict_buffer_checking
        self.per_timestep_loss = per_timestep_loss
        self.ttfs_alpha = ttfs_alpha
        self.softmax_temperature = softmax_temperature
        self._optimiser = get_object(optimiser, Optimiser, "Optimiser",
                                     default_optimisers)
        self._delay_optimiser = get_object(
            optimiser if delay_optimiser is None else delay_optimiser, 
            Optimiser, "Optimiser", default_optimisers)
        self.delay_learn_conns = set(get_underlying_conn(c)
                                     for c in delay_learn_conns)

    def pre_compile(self, network: Network, 
                    genn_model, **kwargs) -> CompileState:
        # Build list of output populations
        readouts = [p for p in network.populations
                    if p.neuron.readout is not None]

        return CompileState(self.losses, readouts,
                            genn_model.backend_name)

    def apply_delay(self, genn_pop, conn: Connection,
                    delay, compile_state):
        # Get max delay
        max_delay_steps = _get_conn_max_delay(conn, delay)
        
        # If maximum delay steps is within 16-bit limit, set max delay steps
        if max_delay_steps > 65535:
            raise NotImplementedError(f"Maximum of {conn.max_delay_steps} "
                                      f"delay steps for Connection "
                                      f"{conn.name} exceeds 65535")
        genn_pop.max_dendritic_delay_timesteps = max_delay_steps

    def build_neuron_model(self, pop: Population,
                           model: Union[AutoNeuronModel, SynapseModel],
                           compile_state: CompileState) -> NeuronModel:
        # Build GeNNCode neuron model implementing forward pass of model
        genn_model = super(EventPropCompiler, self).build_neuron_model(
                           pop, model, compile_state)

        # If population has a readout i.e. it's an output
        if pop.neuron.readout is not None:
            return self._build_out_neuron_model(pop, model, genn_model,
                                                compile_state)
        # Otherwise, it's either an input or a hidden neuron
        else:
            return self._build_in_hid_neuron_model(pop, model, genn_model,
                                                   compile_state)

    def build_synapse_model(self, conn: Connection, 
                            model: Union[AutoSynapseModel, SynapseModel],
                            compile_state: CompileState) -> SynapseModel:
        # Check synapse i
        logger.debug(f"Building synapse model for '{conn.name}'")
        if not isinstance(model, AutoSynapseModel):
            raise NotImplementedError(
                "EventProp compiler only supports "
                "synapses defined in terms of AutoSynapseModel")

        # Get target neuron mdeol
        trg_pop = conn.target()
        trg_neuron_model = trg_pop.neuron.get_model(trg_pop, self.dt,
                                                    self.batch_size)
        assert isinstance(trg_neuron_model, AutoNeuronModel)
    
   
        # Build GeNNCode neuron model implementing forward pass of model
        genn_model = super(EventPropCompiler, self).build_synapse_model(
                           conn, model, compile_state)

        logger.debug("\tBuilding adjoint system for AutoSynapseModel:")
        logger.debug(f"\t\tVariables: {model.var_vals.keys()}")
        logger.debug(f"\t\tParameters: {model.param_vals.keys()}")
        logger.debug(f"\t\tForward ODEs: {model.dx_dt}")
        logger.debug(f"\t\tForward jumps: {model.jumps}")

        # Loop through synapse ODEs
        dl_dt = {}
        isyn_sym = sympy.Symbol("Isyn")
        for syn_sym in model.dx_dt.keys():
            # Add adjoint variable to synapse model
            genn_model.add_var(_get_lmd_name(syn_sym), "scalar", 0.0)
            
            # Differentiate all synapse and target neuron ODEs wrt this 
            # synapse var to obtain adjoint lambda ODE
            o = sum(sympy.diff(syn_expr2, syn_sym) * _get_lmd_sym(syn_sym2)
                    for syn_sym2, syn_expr2 in model.dx_dt.items())
            o += sum(sympy.diff(trg_expr.subs(isyn_sym, model.inject_current), 
                                syn_sym) * _get_lmd_sym(trg_sym)
                     for trg_sym, trg_expr in trg_neuron_model.dx_dt.items())
            
            # Check the that lambda ODE does not end up referencing any 
            err = (any(o.has(syn_sym2) for syn_sym2 in model.dx_dt.keys())
                   or any(o.has(trg_sym) 
                          for trg_sym in trg_neuron_model.dx_dt.keys()))
            if err:
                raise NotImplementedError(
                    f"Equations necessitate saving forward pass variables in a currently not supported setting.")
            
            # If any target population lambda variables are 
            # referenced, add neuron variable references
            for trg_var_name in trg_neuron_model.var_vals.keys():
                lambda_var = _get_lmd_sym(trg_var_name)
                if (o.has(lambda_var) and not genn_model.has_neuron_var_ref(lambda_var.name)):
                    genn_model.add_neuron_var_ref(lambda_var.name, "scalar", lambda_var.name)
            
            # If any target population parameters are 
            # referenced, duplicate in synapse model
            _add_required_parameters(trg_neuron_model, genn_model, o)
    
            # Finally add lambda ODE to adjoint system
            dl_dt[_get_lmd_sym(syn_sym)] = o

        logger.debug(f"\t\tAdjoint ODEs: {dl_dt}")
        logger.debug(f"\t\tReset variables: {genn_model.reset_vars}")

        # Build sim code
        genn_model.prepend_sim_code(
            f"""
            // Backward pass
            {solve_ode(dl_dt, self.solver)}
            """)

        # Add reset logic to reset adjoint state variables 
        # as well as any state variables from the original model
        compile_state.add_synapse_reset_vars(conn, genn_model.reset_vars)
        
        # Return model
        return genn_model

    def build_weight_update_model(self, conn: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state: CompileState) -> WeightUpdateModel:
        logger.debug(f"Building weight update model for '{conn.name}'")

        # Does this connection have a delay?
        has_delay = (not is_value_constant(connect_snippet.delay)
                     or connect_snippet.delay > 0)
        
        # Does this connection have learnable delays
        has_learnable_delay = conn in self.delay_learn_conns

        # Get delay type to use for this connection
        delay_type = get_delay_type(
            _get_conn_max_delay(conn, connect_snippet.delay))

        # Mark which sorts of optimiser connection will require
        compile_state.add_optimiser_connection(conn, connect_snippet.trainable,
                                               has_learnable_delay)

        # Get synapse mdeol
        synapse_model = conn.synapse.get_model(conn, self.dt, self.batch_size)
        assert isinstance(synapse_model, AutoSynapseModel)
        
        # Check validity of synapse model jumps
        # TODO: the jumps that are currently possible to support are essentially where the same
        # function of w is added to all variables that have synaptic jumps. Normally that is just
        # var += w but *could* be var+= w^2 or var += sqrt(w) or whatever.
        # these constraints need to be codified and errors thrown
        for jump_sym, jump_expr in synapse_model.jumps.items():
            if sympy.diff(jump_expr - jump_sym, jump_sym) != 0:
                raise NotImplementedError(
                    "EventProp compiler only supports "
                    "synapses which (only) add input to target variables.")
        
        # Get target neuron mdeol
        trg_pop = conn.target()
        trg_neuron_model = trg_pop.neuron.get_model(trg_pop, self.dt,
                                                    self.batch_size)
        assert isinstance(trg_neuron_model, AutoNeuronModel)

        # Create basic weight update model
        # **TODO** start with _get_delay_weight_update_model if delayed
        genn_model = WeightUpdateModel(
            model=deepcopy(weight_update_model),
            param_vals= {"weight": connect_snippet.weight},
            pre_neuron_var_refs={"BackSpike_pre": "BackSpike"})

        # If weights can be trained
        if connect_snippet.trainable:
            # Ensure weights are instantiated as a state variable
            genn_model.make_param_var("weight")

            # Add weight gradient
            genn_model.add_var("weightGradient", "scalar", 0.0)

            # Add weights to list of checkpoint vars
            compile_state.checkpoint_connection_vars.append((conn, "weight"))
            
            # assemble gradient update
            grad_update = 0
            for jump_sym, jump_expr in synapse_model.jumps.items():
                lambda_sym = _get_lmd_sym(jump_sym)
                grad_update -= lambda_sym * sympy.diff(jump_expr, sympy.Symbol("weight"))
                genn_model.add_psm_var_ref(lambda_sym.name, "scalar", lambda_sym.name)
            
            genn_model.append_pre_event_syn_code(f"weightGradient += {sympy.ccode(grad_update)};")
            logger.debug(f"\tGradient update: {grad_update}")
            logger.debug(f"\tSynapse jumps: {synapse_model.jumps}")
            
            # If any synapse parameters are referenced in gradient 
            # update expression, duplicate in weight update model
            _add_required_parameters(synapse_model, genn_model, grad_update)
            
        # If source neuron isn't an input neuron
        source_neuron = conn.source().neuron
        if not isinstance(source_neuron, Input):
            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

            # assemble dx_dt_plusm; 
            # ***NOTE: we here have to work with the POST-SYNAPTIC neurons 
            # and their equations. This currently only works if the
            # inject_current jumps only depend on weight and inject_current 
            # only enters linearly into rhs of any ODEs
            # Do dx_dt_plusm
            # synaptic ODEs first
            syn_dx_dt_plus_m = {}
            for syn_sym, syn_expr in synapse_model.dx_dt.items():
                syn_expr_plus_m = syn_expr
                for jump_sym, jump_expr in synapse_model.jumps.items():
                    syn_expr_plus_m = syn_expr_plus_m.subs(jump_sym, jump_expr)
                syn_dx_dt_plus_m[syn_sym] = syn_expr_plus_m

            inject_expr = synapse_model.inject_current
            inject_plus_m_expr = inject_expr
            for jump_sym, jump_expr in synapse_model.jumps.items():
                inject_plus_m_expr = inject_plus_m_expr.subs(jump_sym, jump_expr)
            
            isyn_sym = sympy.Symbol("Isyn")
            trg_dx_dt_plus_m = {
                trg_sym: trg_expr.subs(isyn_sym, inject_plus_m_expr)
                for trg_sym, trg_expr in trg_neuron_model.dx_dt.items()}
            
            logger.debug(f"\tSynapse forward ODEs: {synapse_model.dx_dt}")
            logger.debug(f"\tSynapse dx_dtplusm: {syn_dx_dt_plus_m}")
            logger.debug(f"\tTarget neuron forward ODEs: {trg_neuron_model.dx_dt}")
            logger.debug(f"\tTarget neuron n_dx_dtplusm: {trg_dx_dt_plus_m}")

            # SIMPLIFICATON: no dependencies of synaptic jumps on 
            # pre- or post-synaptic variables!
            # add_to_pre is based on the difference of dx_dt and dx_dtplusm 
            # synapse equations first:
            ex = sum(
                _get_lmd_sym(syn_sym) * (syn_dx_dt_plus_m[syn_sym] - syn_expr)
                for syn_sym, syn_expr in synapse_model.dx_dt.items())
            
            # then neuron equations:
            ex += sum(
                _get_lmd_sym(trg_sym) * (trg_dx_dt_plus_m[trg_sym] 
                                        - trg_expr.subs(isyn_sym, inject_expr))
                for trg_sym, trg_expr in trg_neuron_model.dx_dt.items())
            
            if ex != 0:
                add_to_pre = sympy.simplify(ex)
                logger.debug(f"\tAdd to pre: {add_to_pre}")

                # check whether any vars might be involved from the forward pass
                err = (any(add_to_pre.has(sympy.Symbol(s)) 
                        for s in synapse_model.var_vals.keys())
                    or any(add_to_pre.has(sympy.Symbol(s))
                            for s in trg_neuron_model.var_vals.keys()))
                if err:
                    raise NotImplementedError(
                        "Synapse equations which require saving forward "
                        "pass variables are currently not supported.")

                # If any target neuron parameters are referenced in 
                # add to pre expression, duplicate in weight update model
                _add_required_parameters(trg_neuron_model, genn_model, add_to_pre)

                # If any synapse parameters are referenced in add to pre
                # expression, duplicate in weight update model
                _add_required_parameters(synapse_model, genn_model, add_to_pre)

                # If any target population variables or lambda variables are 
                # referenced, add neuron variable references
                for trg_var_name in trg_neuron_model.var_vals.keys():
                    if add_to_pre.has(sympy.Symbol(trg_var_name)):
                        genn_model.add_post_neuron_var_ref(trg_var_name, 
                                                        "scalar", trg_var_name)
                    lambda_var = _get_lmd_sym(trg_var_name)
                    if add_to_pre.has(lambda_var):
                        genn_model.add_post_neuron_var_ref(lambda_var.name, 
                                                        "scalar", lambda_var.name)

                # If any synapse model variables or lambda variables are 
                # referenced, add psm variable references
                for syn_var_name in synapse_model.var_vals.keys():
                    if add_to_pre.has(sympy.Symbol(syn_var_name)):
                        genn_model.add_psm_var_ref(syn_var_name, 
                                                "scalar", syn_var_name)
                    l_var = _get_lmd_sym(syn_var_name)
                    if add_to_pre.has(l_var) and not genn_model.has_psm_var_ref(l_var.name):
                        genn_model.add_psm_var_ref(lambda_var.name, 
                                                "scalar", lambda_var.name)

            # Convert expression to c-code and insert call to addToPre
            genn_model.append_pre_event_syn_code(f"addToPre({sympy.ccode(add_to_pre)});")

        return genn_model

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict, 
                                compile_state: CompileState) -> CompiledTrainingNetwork:
        # Correctly target feedback
        for c in compile_state.feedback_connections:
            connection_populations[c].pre_target_var = "RevISyn"

        # Loop through connections that require optimisers
        weight_optimiser_cus = []
        delay_optimiser_cus = []
        for i, (c, w, d) in enumerate(compile_state.optimiser_connections):
            genn_pop = connection_populations[c]
            
            # If weight optimisation is required
            gradient_vars = []
            if w:
                # Create weight optimiser custom update
                cu_weight = self._create_optimiser_custom_update(
                    f"Weight{i}", create_wu_var_ref(genn_pop, "weight"),
                    create_wu_var_ref(genn_pop, "weightGradient"), 
                    self._optimiser, genn_model)
                
                # Add custom update to list of optimisers
                weight_optimiser_cus.append(cu_weight)

                # Add gradient to list of gradient vars to zero
                gradient_vars.append(("weightGradient", "scalar", 0.0))
            
            # If delay optimiser is required
            if d:
                # Create delay optimiser custom update
                cu_delay = self._create_optimiser_custom_update(
                    f"Delay{i}", create_wu_var_ref(genn_pop, "delay"),
                    create_wu_var_ref(genn_pop, "delayGradient"),
                    self._delay_optimiser, genn_model,
                    (0.0, c.max_delay_steps))

                # Add custom update to list of optimisers
                delay_optimiser_cus.append(cu_delay)
                
                # Add gradient to list of gradient vars to zero
                gradient_vars.append(("delayGradient", "scalar", 0.0))

            # Create reset model for gradient variables
            assert len(gradient_vars) > 0
            zero_grad_model = create_reset_custom_update(
                gradient_vars,
                lambda name: create_wu_var_ref(genn_pop, name))

            # Add custom update
            self.add_custom_update(genn_model, zero_grad_model, 
                                   "ZeroGradient", f"CUZeroConnGradient{i}")

        # Add per-batch softmax custom updates for each population that requires them
        for p, i, o in compile_state.batch_softmax_populations:
            genn_pop = neuron_populations[p]
            self.add_softmax_custom_updates(genn_model, genn_pop,
                                            i, o, "Batch")

        # Add per-timestep softmax custom updates for each population that requires them
        for p, i in enumerate(compile_state.timestep_softmax_populations):
            # Create custom update model to implement 
            # first softmax pass and add to model
            genn_pop = neuron_populations[p]
            self._add_softmax_buffer_custom_updates(genn_model, genn_pop, i)

        # Loop through connections and add custom updates to zero out post
        for i, genn_syn_pop in enumerate(connection_populations.values()):
            self.add_out_post_zero_custom_update(genn_model, genn_syn_pop,
                                                 "ZeroOutPost",
                                                 f"CUZeroOutPost{i}")
        
        # Loop through populations which require spike 
        # count reductions add custom update
        for i, p in enumerate(compile_state.spike_count_populations):
            genn_pop = neuron_populations[p]
            self._create_spike_count_reduce_custom_update(
                genn_model, genn_pop, f"CUReduceSpikeCount{i}")
            
        # Create custom updates to implement variable reset
        compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations,
                                                  connection_populations)

        # Build list of base callbacks
        base_train_callbacks = []
        base_validate_callbacks = []
        if len(weight_optimiser_cus) > 0 or len(delay_optimiser_cus) > 0:
            if self.full_batch_size > 1:
                base_train_callbacks.append(
                    CustomUpdateOnBatchEndNotFirst("GradientBatchReduce"))
            base_train_callbacks.append(
                CustomUpdateOnBatchEndNotFirst("GradientLearn"))
            base_train_callbacks.append(
                CustomUpdateOnFirstBatchEnd("ZeroGradient"))

        # Add callbacks to set Trial extra global parameter 
        # on populations which require it
        for p in compile_state.update_trial_pops:
            base_train_callbacks.append(UpdateTrial(neuron_populations[p]))

        # Add callbacks to zero out post on all connections
        base_train_callbacks.append(
            CustomUpdateOnLastTimestep("ZeroOutPost", self.example_timesteps))
        base_validate_callbacks.append(
            CustomUpdateOnLastTimestep("ZeroOutPost", self.example_timesteps))

        # If softmax calculation is required at end of batch, add callbacks
        if len(compile_state.batch_softmax_populations) > 0:
            base_train_callbacks.append(CustomUpdateOnBatchEnd("BatchSoftmax1"))
            base_train_callbacks.append(CustomUpdateOnBatchEnd("BatchSoftmax2"))
            base_train_callbacks.append(CustomUpdateOnBatchEnd("BatchSoftmax3"))

        # If softmax calculation is required at end of timestep, add callbacks
        if len(compile_state.timestep_softmax_populations) > 0:
            base_train_callbacks.append(CustomUpdateOnTimestepEnd("Softmax1"))
            base_train_callbacks.append(CustomUpdateOnTimestepEnd("Softmax2"))
            base_train_callbacks.append(CustomUpdateOnTimestepEnd("Softmax3"))
        
        # If spike count reduction is required at end of batch, add callback
        if len(compile_state.spike_count_populations) > 0 and self.full_batch_size > 1:
            base_train_callbacks.append(CustomUpdateOnBatchEnd("SpikeCountReduce"))

        # Add reset custom updates
        if compile_state.is_reset_custom_update_required:
            base_train_callbacks.append(CustomUpdateOnBatchBegin("Reset"))
            base_validate_callbacks.append(CustomUpdateOnBatchBegin("Reset"))
        
        # Build list of optimisers and their custom updates
        optimisers = []
        if len(weight_optimiser_cus) > 0:
            optimisers.append((self._optimiser, weight_optimiser_cus))
        if len(delay_optimiser_cus) > 0:
            optimisers.append((self._delay_optimiser, delay_optimiser_cus))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations,
            self.communicator, compile_state.losses,
            self.example_timesteps, base_train_callbacks,
            base_validate_callbacks, optimisers,
            compile_state.checkpoint_connection_vars,
            compile_state.checkpoint_population_vars, True)

    @property
    def regulariser_enabled(self):
        return (self.reg_lambda_lower != 0.0 
                or self.reg_lambda_upper != 0.0)

    def _add_softmax_buffer_custom_updates(self, genn_model, genn_pop, 
                                           input_var_name: str):
        # Create custom update model to implement 
        # first softmax pass and add to model
        softmax_1 = CustomUpdateModel(
            softmax_1_model, {}, {"MaxVal": 0.0},
            {"Val": create_var_ref(genn_pop, input_var_name)})

        genn_softmax_1 = self.add_custom_update(
            genn_model, softmax_1, 
            "Softmax1",
            "CUSoftmax1" + genn_pop.name)

        # Create custom update model to implement 
        # second softmax pass and add to model
        softmax_2 = CustomUpdateModel(
            softmax_2_model, {}, {"SumExpVal": 0.0},
            {"Val": create_var_ref(genn_pop, input_var_name),
             "MaxVal": create_var_ref(genn_softmax_1, "MaxVal")})

        genn_softmax_2 = self.add_custom_update(
            genn_model, softmax_2, 
            "Softmax2",
            "CUSoftmax2" + genn_pop.name)

        # Create custom update model to implement 
        # third softmax pass and add to model
        softmax_3 = CustomUpdateModel(
            {
                "var_refs": [("Val", "scalar", VarAccessMode.READ_ONLY),
                             ("MaxVal", "scalar", VarAccessMode.READ_ONLY),
                             ("SumExpVal", "scalar", VarAccessMode.READ_ONLY),
                             ("RingWriteOffset", "int")],
                "extra_global_param_refs": [("RingOutputLossTerm", "scalar*")],
                "update_code": f"""
                const int ringOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                RingOutputLossTerm[ringOffset + RingWriteOffset]= exp(Val - MaxVal) / SumExpVal;
                RingWriteOffset++;
                """}, 
            {}, {},
            {"Val": create_var_ref(genn_pop, input_var_name),
             "MaxVal": create_var_ref(genn_softmax_1, "MaxVal"),
             "SumExpVal": create_var_ref(genn_softmax_2, "SumExpVal"),
             "RingWriteOffset": create_var_ref(genn_pop, "RingWriteOffset")},
            {},
            {"RingOutputLossTerm": create_egp_ref(genn_pop, "RingOutputLossTerm")})

        self.add_custom_update(
            genn_model, softmax_3, 
            "Softmax3", 
            "CUSoftmax3" + genn_pop.name)

    
    def _create_optimiser_custom_update(self, name_suffix, var_ref, 
                                        gradient_ref, optimiser, genn_model,
                                        clamp_var=None):
        # If batch size is greater than 1
        if self.full_batch_size > 1:
            # Create custom update model to reduce Gradient into a variable 
            reduction_optimiser_model = CustomUpdateModel(
                gradient_batch_reduce_model, {}, {"ReducedGradient": 0.0},
                {"Gradient": gradient_ref})

            # Add GeNN custom update to model
            genn_reduction = self.add_custom_update(
                genn_model, reduction_optimiser_model, 
                "GradientBatchReduce", "CUBatchReduce" + name_suffix)

            # Create optimiser model without gradient zeroing
            # logic, connected to reduced gradient
            reduced_gradient = create_wu_var_ref(genn_reduction,
                                                 "ReducedGradient")
            optimiser_model = optimiser.get_model(reduced_gradient, var_ref,
                                                  False, clamp_var)
        # Otherwise
        else:
            # Create optimiser model with gradient zeroing 
            # logic, connected directly to population
            optimiser_model = optimiser.get_model(gradient_ref, var_ref,
                                                  True, clamp_var)

        # Add GeNN custom update to model
        return self.add_custom_update(genn_model, optimiser_model,
                                      "GradientLearn",
                                      "CUGradientLearn" + name_suffix)
    
    def _create_spike_count_reduce_custom_update(self, genn_model,
                                                 genn_pop, name: str):
        # If batch size is greater than 1
        if self.full_batch_size > 1:
            # Create custom update model to reduce spike count into a variable 
            reduction_optimiser_model = CustomUpdateModel(
                spike_count_batch_reduce_model, {}, {},
                {"SpikeCount": create_var_ref(genn_pop, "SpikeCount"),
                 "SpikeCountBatch": create_var_ref(genn_pop, 
                                                  "SpikeCountBackBatch")})

            # Add GeNN custom update to model
            self.add_custom_update(
                genn_model, reduction_optimiser_model, 
                "SpikeCountReduce", name)
    
    def _build_adjoint_system(self, model: AutoNeuronModel,
                              output: bool):
        logger.debug("\tBuilding adjoint system for AutoNeuronModel:")
        logger.debug(f"\t\tVariables: {model.var_vals.keys()}")
        logger.debug(f"\t\tParameters: {model.param_vals.keys()}")
        logger.debug(f"\t\tForward ODEs: {model.dx_dt}")
        logger.debug(f"\t\tForward jumps: {model.jumps}")

        # generate adjoint ODE
        # assume that neuron variables do not appear in rhs of post-synapse ODEs
        # therefore, we can do these independent of synapse equations
        saved_vars = set()
        dl_dt = {}
        for sym, expr in model.dx_dt.items():
            o = sum(sympy.diff(expr, sym) * _get_lmd_sym(sym2)
                    for sym2 in model.dx_dt.keys())
            
            # collect variables they might need to go into a ring buffer:
            # **TODO** helper
            saved_vars.update({sym2.name for sym2 in model.dx_dt.keys()
                               if o.has(sym2)})
            dl_dt[_get_lmd_sym(sym)] = o
            
        logger.debug(f"\t\tAdjoint ODE: {dl_dt}")

        # threshold condition
        thresold_expr = (sympy.parse_expr(model.model["threshold"],
                                          local_dict=model.symbols)
                         if ("threshold" in model.model 
                             and model.model["threshold"] is not None)
                         else 0)


        # Substitute jumps into ODEs to get post-jump dynamics "\dot{x}^+"
        dx_dt_plus_n = {}
        for sym, expr in model.dx_dt.items():
            expr_plus_n = expr
            for jump_sym, jump_expr in model.jumps.items():
                expr_plus_n = expr_plus_n.subs(jump_sym, jump_expr)

            dx_dt_plus_n[sym] = expr_plus_n

        logger.debug(f"\t\tdx_dtplusn: {dx_dt_plus_n}")

        a = {}
        b = {}
        c = {}
        for var_name in model.var_vals.keys():
            var_sym = sympy.Symbol(var_name)
            a[var_sym] = sympy.simplify(
                sum(sympy.diff(jump_expr, var_sym) * _get_lmd_sym(jump_sym)
                    for jump_sym, jump_expr in model.jumps.items()))

            # **TODO** helper
            saved_vars.update({var_name2 for var_name2 in model.var_vals.keys()
                               if a[var_sym].has(sympy.Symbol(var_name2))})

            if thresold_expr != 0:
                ex = sympy.simplify(
                    sum(sympy.diff(thresold_expr, var_sym2) * var_expr2
                        for var_sym2, var_expr2 in model.dx_dt.items()))
                if ex != 0:
                    ex = sympy.diff(thresold_expr, var_sym) / ex
                    if ex != 0:
                        b[var_sym] = _simplify_using_threshold(
                            model.var_vals.keys(), thresold_expr, ex)

                        # **TODO** helper
                        saved_vars.update(
                            {var_name2 for var_name2 in model.var_vals.keys()
                             if b[var_sym].has(sympy.Symbol(var_name2))})
                if var_sym in b:
                    ex = 0
                    for jump_sym, jump_expr in model.jumps.items():
                        ex2 = sum(
                            sympy.diff(jump_expr, var_sym2) * var_expr2
                            for var_sym2, var_expr2 in model.dx_dt.items())

                        ex2 -= dx_dt_plus_n[jump_sym]
                        ex -= _get_lmd_sym(jump_sym) * ex2
                    ex = sympy.simplify(ex)
                    if ex != 0:
                        c[var_sym] = _simplify_using_threshold(
                            model.var_vals.keys(), thresold_expr, ex)
                        saved_vars.update(
                            {var_name2 for var_name2 in model.var_vals.keys()
                             if c[var_sym].has(sympy.Symbol(var_name2))})

        logger.debug(f"\t\tA: {a}")
        logger.debug(f"\t\tB: {b}")
        logger.debug(f"\t\tC: {c}")

        # assemble the different lambda jump parts
        adjoint_jumps = {}
        for a_sym, a_exp in a.items():
            if a_sym in b and a_sym in c:
                ex2 = c[a_sym]
            else:
                ex2 = 0
            if a_sym in b:
                # TODO: we are missing l_V^- - l_V^+ for output neurons with jumps
                # that are combined with l_V loss types
                # This is at the moment categorically excluded
                drive = sympy.Symbol("drive_p" if output else "RevISyn")
                jump = a_exp + b[a_sym] * (ex2 + drive)
            else:
                jump = a_exp
            jump =  _simplify_using_threshold(model.var_vals.keys(),
                                              thresold_expr, jump)
            
            # If any jumping remains
            if jump != 0:
                # Loop through all forward variables (and Isyn)
                for var_name in chain(model.var_vals.keys(), ["Isyn"]):
                    var_sym = sympy.Symbol(var_name)

                    # If there're referenced in jump, add $ to name so we can 
                    # go through with template engine and replace them all
                    # **THOMAS** are any previous points where things are added to saved_vars required?
                    if jump.has(var_sym):
                        saved_vars.add(var_sym.name)
                        jump = jump.subs(var_sym, sympy.Symbol(f"${var_name}"))
                
                # Add to dictionary as jump for adjoint variable
                adjoint_jumps[_get_lmd_sym(a_sym)] = jump

        logger.debug(f"\t\tAdjoint Jumps: {adjoint_jumps}")
        logger.debug(f"\t\tSaved variables: {saved_vars}")
        
        return dl_dt, adjoint_jumps, saved_vars
    
    def _build_in_hid_neuron_model(self, pop: Population,
                                   model: Union[AutoNeuronModel, NeuronModel],
                                   genn_model: NeuronModel,
                                   compile_state: CompileState) -> NeuronModel:
        # Add variables to hold offsets for 
        # reading and writing ring variables
        genn_model.add_var("RingWriteOffset", "int", 0, reset=False)
        genn_model.add_var("RingReadOffset", "int", self.max_spikes - 1,
                           reset=False)

        # Add variables to hold offsets where this neuron
        # started writing to ring during the forward
        # pass and where data to read during backward pass ends
        genn_model.add_var("RingWriteStartOffset", "int",
                           self.max_spikes - 1, reset=False)
        genn_model.add_var("RingReadEndOffset", "int", 
                           self.max_spikes - 1, reset=False)

        # Add variable to hold backspike flag
        genn_model.add_var("BackSpike", "uint8_t", False)

        # Add EGP for spike time ring variables
        spike_ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
        genn_model.add_egp("RingSpikeTime", "scalar*", 
                           np.empty(spike_ring_size, dtype=np.float32))

        # If neuron is an input
        if isinstance(pop.neuron, Input):
            # Add reset logic to reset any state 
            # variables from the original model
            logger.debug(f"Building input neuron model for '{pop.name}'")
            logger.debug(f"\tReset variables: {genn_model.reset_vars}")
            compile_state.add_neuron_reset_vars(pop, genn_model.reset_vars,
                                                True, False)

            # No additional dynamics, transition or writing code is required
            dynamics_code = ""
            transition_code = ""
            write_code = ""
        # Otherwise i.e. it's hidden
        else:
            logger.debug(f"Building hidden neuron model for '{pop.name}'")
            if not isinstance(model, AutoNeuronModel):
                raise NotImplementedError(
                    "EventProp compiler only supports hidden "
                    "neurons defined in terms of AutoSynapseModel")
            
            # Build adjoint system from model
            dl_dt, adjoint_jumps, saved_vars =\
                self._build_adjoint_system(model, False)
            
            # Generate transition code
            transition_code = "\n".join(
                f"{jump_sym.name} = {sympy.ccode(jump_expr)};"
                for jump_sym, jump_expr in adjoint_jumps.items())
            
            # Substitute saved variables for those in the ring buffer
            transition_code = Template(transition_code).substitute(
                {s: f"Ring{s}[ringOffset + RingReadOffset]" for s in saved_vars})

            # Add additional input variable to receive add_to_pre feedback
            genn_model.add_additional_input_var("RevISyn", "scalar", 0.0)

            # Add EGP for stored vars ring variables
            ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            for var in saved_vars:
                genn_model.add_egp(f"Ring{var}", "scalar*", 
                                   np.empty(ring_size, dtype=np.float32))

            # Add state variables and reset for lambda variables
            for lambda_sym in dl_dt.keys():
                genn_model.add_var(lambda_sym.name, "scalar", 0.0)

            # If regularisation is enabled
            # **THINK** is this LIF-specific?
            additional_reset_vars = []
            if self.regulariser_enabled:
                # Add state variables to hold spike count
                # during forward and backward pass. 
                # **NOTE** SpikeCountBackSum is shared across
                # batches as it is the result of a reduction
                # **NOTE** if batch size > 1, SpikeCountBackBatch is
                # calculated with a reduction which zeroes SpikeCount
                genn_model.add_var("SpikeCount", "int", 0, 
                                   reset=(self.full_batch_size == 1))
                genn_model.add_var("SpikeCountBackBatch", "int", 0,
                                   VarAccess.READ_ONLY, reset=False)

                # Add parameters for regulariser
                # **NOTE** this is multiplied by batch_size so it
                # can be compared directly to SpikeCountBackBatch
                genn_model.add_param("RegNuUpperBatch", "int",
                                     self.reg_nu_upper * self.full_batch_size)
                    
                # **NOTE** these are divided by batch size once to
                # make these batch-size-agnostic and again to take 
                # into account that we're operating on batch sums of spike counts
                genn_model.add_param(
                    "RegLambdaUpper", "scalar",
                    self.reg_lambda_upper / (self.full_batch_size
                                             * self.full_batch_size))
                genn_model.add_param(
                    "RegLambdaLower", "scalar",
                    self.reg_lambda_lower / (self.full_batch_size
                                             * self.full_batch_size))

                # If batch size is 1, add reset variables to copy SpikeCount
                # into SpikeCountBackBatch and zero SpikeCount
                if self.full_batch_size == 1:
                    additional_reset_vars.append(
                        ("SpikeCountBackBatch", "int", "SpikeCount"))

                # Add additional transition code to apply regularisation
                transition_code = f"""
                    {transition_code}
                    if (SpikeCountBackBatch > RegNuUpperBatch) {{
                        {_get_lmd_name(model.output_var_name)} -= RegLambdaUpper * (SpikeCountBackBatch - RegNuUpperBatch);
                    }}
                    else {{
                        {_get_lmd_name(model.output_var_name)} -= RegLambdaLower * (SpikeCountBackBatch - RegNuUpperBatch);
                    }}
                    """
                    
                # Add population to list of those that 
                # require a spike count reduction
                compile_state.spike_count_populations.append(pop)

                # Add code to update SpikeCount in forward reset code
                genn_model.append_reset_code("SpikeCount++;")

            # Add reset logic to reset state variables (including adjoint)
            all_reset_vars = genn_model.reset_vars + additional_reset_vars
            logger.debug(f"\tReset variables: {all_reset_vars}")
            compile_state.add_neuron_reset_vars(
                pop, all_reset_vars, True, False)

            # Generate ring-buffer write code
            write_code ="\n".join(f"Ring{v}[ringOffset + RingWriteOffset] = {v};"
                                  for v in saved_vars)

            # Solve ODE and generate dynamics code
            dynamics_code = solve_ode(dl_dt, self.solver)
            
        # Add code to start of sim code to run 
        # backwards pass and handle back spikes
        genn_model.prepend_sim_code(
            neuron_backward_pass.substitute(
                max_spikes=self.max_spikes,
                example_time=(self.example_timesteps * self.dt),
                dynamics=dynamics_code,
                transition=transition_code))

        # Prepend code to reset to write spike time to ring buffer
        genn_model.prepend_reset_code(
            neuron_reset.substitute(
                max_spikes=self.max_spikes,
                write=write_code,
                strict_check=(neuron_reset_strict_check
                                if self.strict_buffer_checking
                                else "")))
        return genn_model
        
    def _build_out_neuron_model(self, pop: Population, 
                                model: AutoNeuronModel,
                                genn_model: NeuronModel,
                                compile_state: CompileState) -> NeuronModel:
        logger.debug(f"Building output neuron model for '{pop.name}'")

        # Check neuron model is compatible
        if not isinstance(model, AutoNeuronModel):
            raise NotImplementedError(
                "EventProp compiler only supports output neurons "
                "defined in terms of AutoSynapseModel")
                
        # Check loss function is compatible
        # **TODO** categorical crossentropy i.e. one-hot encoded
        pop_loss = compile_state.losses[pop]
        sce_loss = isinstance(pop_loss, SparseCategoricalCrossentropy)
        mse_loss = isinstance(pop_loss, MeanSquareError)
        if not (sce_loss or mse_loss):
            raise NotImplementedError(
                f"EventProp compiler doesn't support "
                f"{type(pop_loss).__name__} loss")

        # Add output logic to model
        pop.neuron.readout.add_readout_logic(
            genn_model, max_time_required=True, dt=self.dt,
            example_timesteps=self.example_timesteps)

        # Build adjoint system from model
        dl_dt, adjoint_jumps, saved_vars =\
            self._build_adjoint_system(model, True)

        # Add continous drive term to LambdaV
        dl_dt[_get_lmd_sym(model.output_var_name)] += sympy.Symbol("drive")

        # Add adjoint state variables
        # **THINK** what about reset
        for lambda_sym in dl_dt.keys():
            genn_model.add_var(lambda_sym.name, "scalar", 0.0)
        
        # Prepend continous adjoint system update
        genn_model.prepend_sim_code(solve_ode(dl_dt, self.solver))
        
        # **TODO** move logic into loss classes
        # **HACK** we don't want to call add_to_neuron on loss function as
        # it will add unwanted code to end of neuron but we do want this
        if sce_loss:
            # Add variable, shared across neurons to hold true label for batch
            genn_model.add_var("YTrue", "uint8_t", 0, 
                               VarAccess.READ_ONLY_SHARED_NEURON, reset=False)

            # Add second variable to hold the true label for the backward pass
            genn_model.add_var("YTrueBack", "uint8_t", 0, 
                               VarAccess.READ_ONLY_SHARED_NEURON, reset=False)
        elif mse_loss:
            # The true label is the desired voltage output over time
            flat_shape = np.prod(pop.shape)
            egp_size = (self.example_timesteps * self.batch_size * flat_shape)
            genn_model.add_egp("YTrue", "scalar*",
                               np.empty(egp_size, dtype=np.float32))

        # Add dynamic parameter to contain trial index and add 
        # population to list of those which require it updating
        genn_model.add_param("Trial", "unsigned int", 0)
        genn_model.set_param_dynamic("Trial")
        compile_state.update_trial_pops.append(pop)

        # If model is non-spiking - MSE and SCE losses of "voltage V" apply
        if "threshold" not in model.model or model.model["threshold"] is None:
            # Check adjoint system is also jump-less
            assert len(saved_vars) == 0
            assert len(adjoint_jumps) == 0

            # If we want to calculate mean-squared error or per-timestep loss
            out_var_name = model.output_var_name
            if self.per_timestep_loss or mse_loss:
                # Add variables to hold offsets for 
                # reading and writing ring variables
                genn_model.add_var("RingWriteOffset", "int", 0, reset=False)
                genn_model.add_var("RingReadOffset", "int", 0, reset=False)

                # Add EGP for softmax V (SCE) or regression difference (MSE) ring variable
                ring_size = self.batch_size * np.prod(pop.shape) * 2 * self.example_timesteps
                genn_model.add_egp("RingOutputLossTerm", "scalar*", 
                                   np.empty(ring_size, dtype=np.float32))
                
                # If sparse cross-entropy loss is selected
                if sce_loss:
                    # If readout is AvgVar or SumVar
                    if isinstance(pop.neuron.readout, (AvgVar, SumVar)):
                        genn_model.prepend_sim_code(
                            f"""
                            const int ringOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                            if (Trial > 0) {{
                                RingReadOffset--;
                                const scalar softmax = RingOutputLossTerm[ringOffset + RingReadOffset];
                                const scalar g = (id == YTrueBack) ? (1.0 - softmax) : -softmax;
                                drive = g / (num_batch * {self.dt * self.example_timesteps});
                                drive_p = 0.0;
                            }}
                            
                            // Forward pass
                            """)

                        # Add custom updates to calculate 
                        # softmax from V and write directly to buffermodel_copy
                        compile_state.timestep_softmax_populations.append(
                            (pop, out_var_name))

                        # Add custom update to reset state
                        compile_state.add_neuron_reset_vars(pop, genn_model.reset_vars,
                                                            False, True)
                    
                    # Otherwise, unsupported readout type
                    else:
                        raise NotImplementedError(
                            f"EventProp compiler with CategoricalCrossEntropy loss doesn't support "
                            f"{type(pop.neuron.readout).__name__} readouts")
                elif mse_loss:
                    # Readout has to be Var
                    if isinstance(pop.neuron.readout, Var):
                        genn_model.prepend_sim_code(
                            f"""
                            const int ringOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                            if (Trial > 0) {{
                                RingReadOffset--;
                                const scalar error = RingOutputLossTerm[ringOffset + RingReadOffset];
                                drive = error / (num_batch * {self.dt * self.example_timesteps});
                                drive_p  = 0.0;
                            }}
                            """)

                        # Add custom update to reset state JAMIE_CHECK
                        compile_state.add_neuron_reset_vars(
                            pop, genn_model.reset_vars, False, True)

                        # Add code to fill errors into RingBuffer
                        genn_model.append_sim_code(
                            f"""
                            const unsigned int timestep = (int)round(t / dt);
                            const unsigned int index = (batch * {self.example_timesteps} * num_neurons)
                            + (timestep * num_neurons) + id;
                            RingOutputLossTerm[ringOffset + RingWriteOffset] = YTrue[index] - {out_var_name};
                            RingWriteOffset++;
                            """)
                    # Otherwise, unsupported readout type
                    else:
                        raise NotImplementedError(
                            f"EventProp compiler with MeanSqareError loss only supports "
                            f"'Var' readouts") 
            # Otherwise, we want to calculate loss over each trial
            else:
                if sce_loss:
                    # Add state variable to hold softmax of output
                    genn_model.add_var("Softmax", "scalar", 0.0,
                                       VarAccess.READ_ONLY_DUPLICATE,
                                       reset=False)

                    # If readout is AvgVar or SumVar
                    if isinstance(pop.neuron.readout, (AvgVar, SumVar)):
                        genn_model.prepend_sim_code(
                            f"""
                            if (Trial > 0) {{
                                const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                                drive = g / (num_batch * {self.dt * self.example_timesteps});
                                drive_p = 0.0;
                            }}
                            
                            // Forward pass
                            """)
                    
                        # Add custom updates to calculate 
                        # softmax from VSum or VAvg
                        suffix = ("Sum" if isinstance(pop.neuron.readout, SumVar)
                                  else "Avg")
                        compile_state.batch_softmax_populations.append(
                            (pop, out_var_name + suffix, "Softmax"))

                        # Add custom update to reset state
                        compile_state.add_neuron_reset_vars(
                            pop, genn_model.reset_vars, False, False)
                    # Otherwise, if genn_model is AvgVarExpWeight
                    elif isinstance(pop.neuron.readout, AvgVarExpWeight):
                        local_t_scale = 1.0 / (self.dt * self.example_timesteps)
                        genn_model.prepend_sim_code(
                            f"""
                            if (Trial > 0) {{
                                const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                                drive = (g * exp(-(1.0 - (t * {local_t_scale})))) / (num_batch * {self.dt * self.example_timesteps});
                                drive_p = 0.0;
                            }}
                            
                            // Forward pass
                            """)
                    
                        # Add custom updates to calculate softmax from VAvg
                        compile_state.batch_softmax_populations.append(
                            (pop, out_var_name + "Avg", "Softmax"))

                        # Add custom update to reset state
                        compile_state.add_neuron_reset_vars(
                            pop, genn_model.reset_vars, False, False)
                    # Otherwise, if readout is MaxVar
                    elif isinstance(pop.neuron.readout, MaxVar):
                        # Add state variable to hold vmax from previous trial
                        genn_model.add_var(model.output_var_name + "MaxTimeBack", "scalar", 0.0,
                                           VarAccess.READ_ONLY_DUPLICATE, reset=False)

                        genn_model.prepend_sim_code(
                            f"""
                            if (Trial > 0 && fabs(backT - {out_var_name}MaxTimeBack) < 1e-3*dt) {{
                                const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                                drive = g / (num_batch * {self.dt * self.example_timesteps});
                                drive_p = 0.0;
                            }}
                        
                            // Forward pass
                            """)
                    
                        # Add custom updates to calculate softmax from VMax
                        compile_state.batch_softmax_populations.append(
                            (pop, model.output_var_name + "Max", "Softmax"))

                        # Add custom update to reset state
                        # **NOTE** reset VMaxTimeBack first so VMaxTime isn't zeroed
                        # **TODO** time type
                        compile_state.add_neuron_reset_vars(
                            pop, 
                            [(f"{out_var_name}MaxTimeBack", "scalar", 
                              f"{out_var_name}MaxTime")] + genn_model.reset_vars,
                            False, False)
                    # Otherwise, unsupported readout type
                    else:
                        raise NotImplementedError(
                            f"EventProp compiler doesn't support "
                            f"{type(pop.neuron.readout).__name__} readouts")
                elif mse_loss:
                    raise NotImplementedError(
                        f"EventProp compiler doesn't support "
                        f"time averaged loss for regression.")
            # Prepend standard code to update LambdaV and LambdaI
            genn_model.prepend_sim_code(
                f"""
                // Backward pass
                scalar drive = 0.0;
                scalar drive_p = 0.0;
                """)

            # Add second reset custom update to reset YTrueBack to YTrue
            # **NOTE** seperate as these are SHARED_NEURON variables
            if sce_loss:
                compile_state.add_neuron_reset_vars(
                    pop, [("YTrueBack", "uint8_t", "YTrue")], False, False)
        else:
            # Generate transition code
            transition_code = "\n".join(
                f"{jump_sym.name} = {sympy.ccode(jump_expr)};"
                for jump_sym, jump_expr in adjoint_jumps.items())
            
            # Substitute saved variables for those in the ring buffer
            transition_code = Template(transition_code).substitute(
                {s: f"Ring{s}[ringOffset + RingReadOffset]" for s in saved_vars})

            # Add variables to hold offsets for 
            # reading and writing ring variables
            genn_model.add_var("RingWriteOffset", "int", 0, reset=False)
            genn_model.add_var("RingReadOffset", "int", self.max_spikes - 1,
                               reset=False)

            # Add variables to hold offsets where this neuron
            # started writing to ring during the forward
            # pass and where data to read during backward pass ends
            genn_model.add_var("RingWriteStartOffset", "int",
                                self.max_spikes - 1, reset=False)
            genn_model.add_var("RingReadEndOffset", "int", 
                                self.max_spikes - 1, reset=False)

            # Add variable to hold backspike flag
            genn_model.add_var("BackSpike", "uint8_t", False)

            # Add EGP for spike time ring variables
            spike_ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            genn_model.add_egp("RingSpikeTime", "scalar*", 
                                np.empty(spike_ring_size, dtype=np.float32))
            
            # Add EGP for stored vars ring variables
            ring_size = self.batch_size * np.prod(pop.shape) * 2 * self.example_timesteps
            for var in saved_vars:
                genn_model.add_egp(f"Ring{var}", "scalar*", 
                                   np.empty(ring_size, dtype=np.float32))

            # Add parameters with synaptic decay and scale constants
            #model_copy.add_param("IsynScale", "scalar",
            #    self.dt / (tau_syn  * (1.0 - beta)))
        
            # Readout has to be FirstSpikeTime
            if isinstance(pop.neuron.readout, FirstSpikeTime):
                if not sce_loss:
                    raise NotImplementedError(
                        f"EventProp compiler only supports calculating "
                        f"cross-entropy loss with 'FirstSpikeTime' readouts.")
                
                # **THOMAS** where does this madness come from?
                example_time = self.dt * self.example_timesteps
                phantom_scale = self.ttfs_alpha / (0.0001 * example_time * example_time)
                
                # Add state variable to hold softmax of output
                genn_model.add_var("Softmax", "scalar", 0.0,
                                    VarAccess.READ_ONLY_DUPLICATE, reset=False)
                
                # Add state variable to hold TFirstSpike from previous trial
                # **YUCK** REALLY should be timepoint but then you can't softmax
                genn_model.add_var("TFirstSpikeBack", "scalar", 0.0,
                                    VarAccess.READ_ONLY_DUPLICATE, reset=False)
                
                # In backward pass, update lambdas and apply phantom spike if:
                # 1) This is first timestep of backward pass
                # 2) No spike occurred in preceding forward pass
                # 4) This is correct output neuron 
                dynamics_code = f"""
                    if (Trial > 0 && t == 0.0 && TFirstSpikeBack < -{example_time} && id == YTrueBack) {{
                        drive_p = {phantom_scale / self.batch_size};
                    }}
                    """

                # On backward pass transition, update LambdaV if this is the first spike
                # **THOMAS** why are we dividing by what looks like softmax temperature?
                # **TODO** build transition_code from adjoint_jumps here
                transition_code = f"""
                    if (fabs(backT + TFirstSpikeBack) < 1e-3*dt) {{
                        if (id == YTrueBack) {{
                            const scalar fst = {1.01 * example_time} + TFirstSpikeBack;
                            drive_p = (((1.0 - Softmax) / {self.softmax_temperature}) + ({self.ttfs_alpha} / (fst * fst))) / {self.batch_size};
                        }}
                        else {{
                            drive_p = - Softmax / ({self.softmax_temperature * self.batch_size});
                        }}
                    }}
                    {transition_code}
                    """

                # Add reset logic to reset adjoint state variables 
                # as well as any state variables from the original model
                compile_state.add_neuron_reset_vars(
                    pop, [("TFirstSpikeBack", "scalar", "TFirstSpike")] + genn_model.reset_vars,
                    True, False)
                
                # Add second reset custom update to reset YTrueBack to YTrue
                # **NOTE** seperate as these are SHARED_NEURON variables
                compile_state.add_neuron_reset_vars(
                    pop, [("YTrueBack", "uint8_t", "YTrue")], False, False)
        
                # Add code to start of sim code to run backwards pass 
                # and handle back spikes with correct LIF dynamics
                genn_model.prepend_sim_code(
                    neuron_backward_pass.substitute(
                        max_spikes=self.max_spikes,
                        example_time=(self.example_timesteps * self.dt),
                        dynamics=dynamics_code,
                        transition=transition_code))

                # Prepend code defining the drive vars
                genn_model.prepend_sim_code(
                    f"""
                    // Backward pass
                    scalar drive = 0.0;
                    scalar drive_p = 0.0;
                    """)

                # Generate ring-buffer write code
                write_code ="\n".join(f"Ring{v}[ringOffset + RingWriteOffset] = {v};"
                                    for v in saved_vars)

                # Prepend (as it accesses the pre-reset value of V) 
                # code to reset to write spike time and saved vars to ring buffer
                genn_model.prepend_reset_code(
                    neuron_reset.substitute(
                        max_spikes=self.max_spikes,
                        write= write_code,
                        strict_check=(neuron_reset_strict_check 
                                      if self.strict_buffer_checking
                                      else "")))
                
                # Add custom updates to calculate softmax from TFirstSpike
                compile_state.batch_softmax_populations.append(
                    (pop, "TFirstSpike", "Softmax"))
            # Otherwise, unsupported readout type
            else:
                raise NotImplementedError(
                    f"EventProp compiler with spiking output "
                    f"neurons only supports 'FirstSpikeTime' readouts")

        return genn_model