import logging
import numpy as np
import sympy

from abc import ABC
from string import Template
from typing import Mapping, Union, Tuple
from pygenn import (CustomUpdateVarAccess, SynapseMatrixType,
                    VarAccess, VarAccessMode)
from .compiler import Compiler
from .compiled_training_network import CompiledTrainingNetwork
from .ground_truths import GroundTruth
from .. import Connection, InputLayer, Layer, Population, Network
from ..callbacks import (Callback, CustomUpdateOnBatchBegin,
                         CustomUpdateOnBatchEnd, CustomUpdateOnTimestepEnd)
from ..communicators import Communicator
from ..connection import Connection
from ..losses import (Loss, MeanSquareError, PerNeuronMeanSquareError,
                      RelativeMeanSquareError, SparseCategoricalCrossentropy)
from ..neurons import Input
from ..optimisers import Optimiser
from ..readouts import (AvgVar, AvgVarExpWeight, FirstSpikeTime,
                        EndVar, MaxVar, SumVar, Var)
from ..utils.auto_model import AutoModel, AutoNeuronModel, AutoSynapseModel
from ..utils.model import (CustomUpdateModel, Model, NeuronModel, 
                           SynapseModel, WeightUpdateModel)
from ..utils.snippet import ConnectivitySnippet

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from warnings import warn
from pygenn import (create_egp_ref, create_psm_var_ref,
                    create_var_ref, create_wu_var_ref)
from pygenn._genn import WUVarReference
from .compiler import (create_reset_custom_update, get_delay_type,
                       get_conn_max_delay)
from ..utils.auto_tools import solve_ode
from ..utils.module import get_object, get_object_mapping
from ..utils.value import is_value_constant

from .compiler import softmax_1_model, softmax_2_model
from .ground_truths import default_ground_truths
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

learnable_delay_weight_update_model = {
    "params": [("weight", "scalar"), ("delay", "scalar"),
               ("MaxDelay", "int")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
                             
    "pre_spike_syn_code": """
    const int delayInt = max(0, min(MaxDelay, (int)round(delay)));
    addToPostDelay(weight, delayInt);
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
    "var_refs": [("SpikeCount", "int", VarAccessMode.READ_ONLY),
                 ("SpikeCountBatch", "int", VarAccessMode.REDUCE_SUM)],
    "update_code": """
    SpikeCountBatch = SpikeCount;
    """}

abs_sum_reduce_batch_model = {
    "vars": [("BRedAbsSum", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
    "var_refs": [("AbsSum", "scalar",VarAccessMode.READ_ONLY)],
    "update_code": """
    BRedAbsSum = AbsSum;
    """}

abs_sum_reduce_neuron_model_assign  = {
    "params": [("timesteps", "int"),("gradLimit","scalar")],
    "var_refs": [("BRedAbsSum", "scalar", VarAccessMode.READ_ONLY),
                 ("Limit", "scalar", VarAccessMode.REDUCE_SUM)],
    "update_code": """
    Limit = gradLimit*BRedAbsSum/timesteps/num_batch/num_neurons;
    """}

# Template used to generate backward passes for neurons
neuron_backward_pass = Template(
    """
    const int ringOffset = (batch * num_neurons * $max_spikes) + (id * $max_spikes);
    $tsringoffset
    const scalar backT = $example_time - t - dt;

    // Backward pass
    $write
    if (Trial > 0) {
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
        if (RingReadOffset != RingReadEndOffset && (backT - RingSpikeTime[ringOffset + RingReadOffset] - dt) <= 0.1*dt) {
            BackSpike = true;
        }
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
            addToPostDelay(weight, delay);
            """,
            "pre_event_threshold_condition_code": """
            BackSpike_pre
            """}

def _template_symbol(expression, symbol: sympy.Symbol):
    return expression.subs(symbol, sympy.Symbol(f"${symbol.name}"))

def _template_symbols(expression, sym_names, referenced_names: set):
    for n in sym_names:
        sym = sympy.Symbol(n)

        # If there're referenced in jump, add $ to name so we can 
        # go through with template engine and replace them all
        # **THOMAS** are any previous points where things are added to saved_vars required?
        if expression.has(sym):
            referenced_names.add(n)
            expression = _template_symbol(expression, sym)
    return expression

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

def _add_required_parameters(model: AutoModel, genn_model: Model, expression,
                             learn_params, add_var_ref_fn, has_var_ref_fn):
    # Loop through parameters in model
    for n, v in model.param_vals.items():
        # If they are referenced in expression
        if (expression.has(sympy.Symbol(n))):
            # If their value is constant and parameter doesn't already exist,
            # duplicate value in another parameter
            if is_value_constant(v) and n not in learn_params:
                if not genn_model.has_param(n):
                    genn_model.add_param(n, "scalar", v)
            # Otherwise, if variable reference doesn't already exist, 
            # Add suitable read-only variable reference
            elif not has_var_ref_fn(genn_model, n):
                add_var_ref_fn(genn_model, n, "scalar", n,
                               VarAccessMode.READ_ONLY)

def _add_required_psm_neuron_parameters(model: AutoNeuronModel, 
                                        genn_model: SynapseModel,
                                        expression, learn_params):
    _add_required_parameters(model, genn_model, expression, learn_params,
                             SynapseModel.add_neuron_var_ref, 
                             SynapseModel.has_neuron_var_ref)

def _add_required_wum_post_neuron_parameters(model: AutoNeuronModel, 
                                             genn_model: WeightUpdateModel, 
                                             expression, learn_params):
    _add_required_parameters(model, genn_model, expression, learn_params,
                             WeightUpdateModel.add_post_neuron_var_ref,
                             WeightUpdateModel.has_post_neuron_var_ref)

def _add_required_wum_psm_parameters(model: AutoSynapseModel, 
                                     genn_model: WeightUpdateModel, 
                                     expression):
    _add_required_parameters(model, genn_model, expression, {},
                             WeightUpdateModel.add_psm_var_ref,
                             WeightUpdateModel.has_psm_var_ref)


class CompileState:
    def __init__(self, network: Network, losses, optimisers, 
                 supported_matrix_type, backend_name):
        self.backend_name = backend_name
        self._neuron_reset_vars = []
        self._synapse_reset_vars = []
        self.spike_count_populations = []
        self.ttfs_reduce_populations = []
        self.batch_softmax_populations = []
        self.timestep_softmax_populations = []
        self.feedback_connections = []
        self.update_trial_pops = []
        self.adjoint_limit_pops_vars = []
        self.optimisers = {}
        self.loss_recorder_populations = []

        # Build list of output populations
        readouts = [p for p in network.populations
                    if p.neuron.readout is not None]

        # From these, create losses
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)

        # And build matching dictionary of ground truth
        # values each loss function requires 
        self.ground_truths = {k: get_object(l.ground_truth, GroundTruth, 
                                            "GroundTruth", default_ground_truths)
                              for k, l in self.losses.items()}

        # If default optimiser settings for all connections has been provided
        if "all_connections" in optimisers:
            # Loop through all connections
            vars = optimisers["all_connections"]
            for conn in network.connections:
                # If connectivity is trainable, create 
                # optimisers for specified variables
                connect_snippet = conn.connectivity.get_snippet(
                    conn, supported_matrix_type)
                if connect_snippet.trainable:
                    self.optimisers[conn] = {n: get_object(o, Optimiser, 
                                                           "Optimiser",
                                                           default_optimisers)
                                             for n, o in vars.items()}

        # Loop through optimisers to build pre-processed dictionary
        # **NOTE** these will override any optimiser configured with shortcuts
        for k, vars in optimisers.items():
            # If key is a Connection, Population or InputLayer,
            # what variables relate to is unambiguous
            if isinstance(k, (Connection, Population, InputLayer)):
                # Create optimisers
                vars = {n: get_object(o, Optimiser, "Optimiser",
                                      default_optimisers)
                        for n, o in vars.items()}
                
                # If key is InputLayer, de-sugar to population
                if isinstance(k, InputLayer):
                    self.optimisers[k.population()] = vars
                # Otherwise, use key directly
                else:
                    self.optimisers[k] = vars
            # Otherwise, if it's a layer, variable might be related
            # to connection OR population contained within layer
            elif isinstance(k, Layer):
                # Split variable dictionary into connection and
                # population variables and create optimisers
                con_vars = {n: get_object(o, Optimiser, "Optimiser",
                                          default_optimisers)
                            for n, o in vars.items()
                            if n == "weight" or n == "delay"}
                pop_vars = {n: get_object(o, Optimiser, "Optimiser",
                                          default_optimisers)
                            for n, o in vars.items()
                            if n != "weight" and n != "delay"}

                # If any of either type of variable exist, add to
                # dictionary with appropriately de-sugared key
                if len(con_vars) > 0:
                    self.optimisers[k.connection()] = con_vars
                if len(pop_vars) > 0:
                    self.optimisers[k.population()] = pop_vars
    
            # Otherwise, if key isn't one of the shortcut strings
            # which have already been processed, give error
            elif k != "all_connections":
                raise RuntimeError(f"Invalid key '{k}' used in 'optimisers' "
                                   f"dictionary. Valid keys are Connection, "
                                   f"Population, InputLayer or Layer objects "
                                   f"Or strings such as 'all_connections'")

    def add_neuron_reset_vars(self, pop, reset_vars, 
                              reset_event_ring, reset_v_ring):
        if len(reset_vars) > 0 or reset_event_ring or reset_v_ring:
            self._neuron_reset_vars.append((pop, reset_vars, 
                                            reset_event_ring, reset_v_ring))
    
    def add_synapse_reset_vars(self, conn, reset_vars):
        if len(reset_vars) > 0:
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
            if r_v:
                # Add references to ring buffer offsets
                model.add_var_ref("tsRingReadOffset", "int",
                                  create_var_ref(neuron_pops[pop],
                                                 "tsRingReadOffset"))
                model.add_var_ref("tsRingWriteOffset", "int", 
                                  create_var_ref(neuron_pops[pop],
                                                 "tsRingWriteOffset"))
                # Add additional update code to update ring buffer offsets
                model.append_update_code(
                    f"""
                    tsRingReadOffset = tsRingWriteOffset;
                    if (tsRingWriteOffset >= {2 * compiler.example_timesteps}) {{
                        tsRingWriteOffset = 0;
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
    def is_reset_custom_update_required(self):
        return (len(self._neuron_reset_vars) > 0
                or len(self._synapse_reset_vars) > 0)


class UpdateTrial(Callback):
    def __init__(self, genn_pop):
        self.genn_pop = genn_pop

    def on_batch_begin(self, state, batch: int):
        logger.debug(f"Updating trial at start of batch {batch}")

        # Set dynamic parameter to batch ID
        self.genn_pop.set_dynamic_param_value("Trial", batch)


class CustomUpdateOnLastTimestep(Callback):
    """Callback that triggers a GeNN custom update 
    at the start of the last timestep in each example"""
    def __init__(self, name: str, example_timesteps: int):
        self.name = name
        self.example_timesteps = example_timesteps
    
    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_timestep_begin(self, state, timestep: int):
        if timestep == (self.example_timesteps - 1):
            logger.debug(f"Running custom update {self.name} "
                         f"at start of timestep {timestep}")
            state.genn_model.custom_update(self.name)


class CustomUpdateOnBatchEndNotFirst(Callback):
    """Callback that triggers a GeNN custom update 
    at the end of every batch after the first."""
    def __init__(self, name: str):
        self.name = name

    def create_state(self, compiled_network, **kwargs):
        return compiled_network
        
    def on_batch_end(self, state, batch, metric_state):
        if batch > 0:
            logger.debug(f"Running custom update {self.name} "
                         f"at end of batch {batch}")
            state.genn_model.custom_update(self.name)

class CustomUpdateOnFirstBatchEnd(Callback):
    """Callback that triggers a GeNN custom update 
    at the end of first batch."""
    def __init__(self, name: str):
        self.name = name

    def create_state(self, compiled_network, **kwargs):
        return compiled_network
        
    def on_batch_end(self, state, batch, metric_state):
        if batch == 0:
            logger.debug(f"Running custom update {self.name} "
                         f"at end of batch {batch}")
            state.genn_model.custom_update(self.name)

@dataclass
class LossRecorderState:
    compiled_network: CompiledTrainingNetwork
    losses: list = field(default_factory=list)

class LossRecorderCallback(Callback, ABC):
    def __init__(self, pop: Population, key: str):
        self.pop = pop
        self.key = key

    def create_state(self, compiled_network, **kwargs):
        return LossRecorderState(compiled_network)

    def on_batch_end(self, state, batch, metric_state):
        # Pull variable loss is calculated from from device
        cn = state.compiled_network
        genn_pop = cn.neuron_populations[self.pop]
        genn_pop.vars["LossSum"].pull_from_device()

        # Calculate loss and add to list
        loss = np.sum(genn_pop.vars["LossSum"].view
                      / cn.genn_model.batch_size)
        state.losses.append(loss)

    def get_data(self, state):
        return self.key, state.losses

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
        reg_lambda:                 Regularisation strength, single value or tuple,
                                    if tuple, (strength for undershoot of hidden
                                    spike number, strength for overshoot)
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
        
    """

    def __init__(self, example_timesteps: int, losses,
                 reg_lambda: Union[float, Tuple[float, float]] = 0.0, reg_nu_upper: float = 0.0,
                 grad_limit: float = 100.0,
                 max_spikes: int = 500, 
                 strict_buffer_checking: bool = False, 
                 per_timestep_loss: bool = False, dt: float = 1.0,
                 ttfs_alpha: float = 0.01, softmax_temperature: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 communicator: Communicator = None,
                 **genn_kwargs):
        supported_matrix_types = [SynapseMatrixType.TOEPLITZ,
                                  SynapseMatrixType.PROCEDURAL_KERNELG,
                                  SynapseMatrixType.DENSE,
                                  SynapseMatrixType.SPARSE]
        
        if "optimiser" in genn_kwargs or "delay_optimiser" in genn_kwargs:
            raise RuntimeError("The 'optimiser' and 'delay_optimiser' "
                               "parameters have been removed from the "
                               "EventPropCompiler constructor. Optimisers "
                               "are now specified by passing a 'optimisers' "
                               "keyword argument to the ``compile`` method "
                               "e.g. optimisers={\"all_connections\": "
                               "{\"weight\": \"adam\"} to optimise all "
                               "weights with the adam optimiser")

        # If regularisation strength is specified as a single float, use for both upper and lower
        if isinstance(reg_lambda, float):
            self.reg_lambda_lower = reg_lambda
            self.reg_lambda_upper = reg_lambda
        # Otherwise, unpack
        else:
            self.reg_lambda_lower, self.reg_lambda_upper = reg_lambda

        # Handle legacy regularisation strength definitions
        reg_warning = False
        if "reg_lambda_lower" in genn_kwargs:
            self.reg_lambda_lower = genn_kwargs.pop("reg_lambda_lower")
            reg_warning = True
        if "reg_lambda_upper" in genn_kwargs:
            self.reg_lambda_upper = genn_kwargs.pop("reg_lambda_upper")
            reg_warning = True
        if reg_warning:
             warn("Seperate 'reg_lambda_upper' and 'reg_lambda_lower' "
                  "arguments for EventPropCompiler are no longer "
                  "supported, please use 'reg_lambda' and use a "
                  "tuple if separate values for undershoot "
                  "and overshoot are required.", FutureWarning)

        super().__init__(supported_matrix_types, dt, batch_size, rng_seed,
                         kernel_profiling, communicator, **genn_kwargs)

        self.example_timesteps = example_timesteps
        self.losses = losses
        self.reg_nu_upper = reg_nu_upper
        self.grad_limit = grad_limit
        self.max_spikes = max_spikes
        self.strict_buffer_checking = strict_buffer_checking
        self.per_timestep_loss = per_timestep_loss
        self.ttfs_alpha = ttfs_alpha
        self.softmax_temperature = softmax_temperature
        

    def pre_compile(self, network: Network, 
                    genn_model, **kwargs) -> CompileState:
        # Get base dictionary of optimiser. If none is provided, default
        # to training all weights using the adam optimiser with default params
        optimisers = kwargs.get("optimisers", 
                                {"all_connections": {"weight": "adam"}})

        # Check dictionary has been provided
        if not isinstance(optimisers, Mapping):
            raise RuntimeError("optimisers should be "
                               "specified as a dictionary")


        return CompileState(network, self.losses, optimisers,
                            self.supported_matrix_type, 
                            genn_model.backend_name)

    def apply_delay(self, genn_pop, conn: Connection,
                    delay, compile_state):
        # Get max delay
        max_delay_steps = get_conn_max_delay(conn, delay)
        
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
        genn_model = super().build_neuron_model(pop, model, compile_state)

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

        # Get target neuron model
        trg_pop = conn.target()
        trg_neuron_model = trg_pop.neuron.get_model(trg_pop, self.dt,
                                                    self.batch_size)
        assert isinstance(trg_neuron_model, AutoNeuronModel)
    
   
        # Build GeNNCode neuron model implementing forward pass of model
        genn_model = super().build_synapse_model(conn, model, compile_state)

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
                    "Synapse equations which require saving forward "
                    "pass variables are currently not supported.")
            
            # If any target population lambda variables are 
            # referenced, add neuron variable references
            for trg_var_name in trg_neuron_model.var_vals.keys():
                lambda_var = _get_lmd_sym(trg_var_name)
                if (o.has(lambda_var) and not genn_model.has_neuron_var_ref(lambda_var.name)):
                    genn_model.add_neuron_var_ref(lambda_var.name, "scalar", lambda_var.name)
            
            # If any target population parameters are 
            # referenced, duplicate in synapse model
            _add_required_psm_neuron_parameters(
                trg_neuron_model, genn_model, o,
                compile_state.optimisers.get(trg_pop, {}))
    
            # Finally add lambda ODE to adjoint system
            dl_dt[_get_lmd_sym(syn_sym)] = o

        logger.debug(f"\t\tAdjoint ODEs: {dl_dt}")
        logger.debug(f"\t\tReset variables: {genn_model.reset_vars}")

        # Build sim code
        genn_model.prepend_sim_code(
            f"""
            // Backward pass
            {solve_ode(dl_dt, model.solver, model.sub_steps)}
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
        has_learnable_delay = (conn in compile_state.optimisers
                               and "delay" in compile_state.optimisers[conn])

        # Get name of variable containing integer delay for indexing
        int_delay_name = "delayInt" if has_learnable_delay else "delay"

        # Get synapse model
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
        
        # Build post-jump synapse ODEs by replacing all variables
        # with jumps in the ODE expressions with jump expressions
        syn_dx_dt_plus_m = {}
        for syn_sym, syn_expr in synapse_model.dx_dt.items():
            syn_expr_plus_m = syn_expr
            for jump_sym, jump_expr in synapse_model.jumps.items():
                syn_expr_plus_m = syn_expr_plus_m.subs(jump_sym, jump_expr)
            syn_dx_dt_plus_m[syn_sym] = syn_expr_plus_m
        
        # Build post-jump injection expression by replacing 
        # all variables with jumps with jump expressions
        inject_plus_m_expr = synapse_model.inject_current
        for jump_sym, jump_expr in synapse_model.jumps.items():
            inject_plus_m_expr = inject_plus_m_expr.subs(jump_sym, jump_expr)

        logger.debug(f"\tSynapse forward ODEs: {synapse_model.dx_dt}")
        logger.debug(f"\tSynapse jumps: {synapse_model.jumps}")
        logger.debug(f"\tSynapse inject_plusm: {inject_plus_m_expr}")
        logger.debug(f"\tSynapse dx_dtplusm: {syn_dx_dt_plus_m}")
        
        # Get target neuron model
        trg_pop = conn.target()
        trg_neuron_model = trg_pop.neuron.get_model(trg_pop, self.dt,
                                                    self.batch_size)
        assert isinstance(trg_neuron_model, AutoNeuronModel)

        # Build post-injection neuron ODEs by replacing Isyn
        # summed input variable with post-jump injection expression
        isyn_sym = sympy.Symbol("Isyn")
        trg_dx_dt_plus_m = {
            trg_sym: trg_expr.subs(isyn_sym, inject_plus_m_expr)
            for trg_sym, trg_expr in trg_neuron_model.dx_dt.items()}
        
        logger.debug(f"\tTarget neuron forward ODEs: {trg_neuron_model.dx_dt}")
        logger.debug(f"\tTarget neuron n_dx_dtplusm: {trg_dx_dt_plus_m}")
        
        #---------------------------------------------------------------------
        # Step 1: create basic GeNN model
        #---------------------------------------------------------------------
        # If connection has learnable delays
        if has_learnable_delay:
            # Check connectivity is trainable
            if not connect_snippet.trainable:
                raise RuntimeError(f"Connection {conn.name} delays cannot be "
                                   f"learned - connectivity is not trainable")

            # Check maximum delay steps is set
            if conn.max_delay_steps is None:
                raise RuntimeError(f"Maximum delay steps must be specified for "
                                   f"Connection {conn.name} with delay learning")

            # Create weight update model
            genn_model = WeightUpdateModel(
                model=deepcopy(learnable_delay_weight_update_model),
                param_vals= {"weight": connect_snippet.weight,
                             "delay": connect_snippet.delay,
                             "MaxDelay": conn.max_delay_steps},
                pre_neuron_var_refs={"BackSpike_pre": "BackSpike"})
        # Otherwise, if connection has static delays
        elif has_delay:
            # Get delay type to use for this connection
            delay_type = get_delay_type(get_conn_max_delay(
                conn, connect_snippet.delay))

            # Create weight update model with delay
            genn_model = WeightUpdateModel(
                model=_get_delay_weight_update_model(delay_type),
                param_vals= {"weight": connect_snippet.weight,
                             "delay": connect_snippet.delay},
                pre_neuron_var_refs={"BackSpike_pre": "BackSpike"})
        # Otherwise, just create basic weight update model
        else:
            genn_model = WeightUpdateModel(
                model=deepcopy(weight_update_model),
                param_vals= {"weight": connect_snippet.weight},
                pre_neuron_var_refs={"BackSpike_pre": "BackSpike"})

        #---------------------------------------------------------------------
        # Step 2: derive dx_dt_diff_sum 
        #---------------------------------------------------------------------
        # If connection has learnable delays or propagates gradients
        source_neuron = conn.source().neuron
        if has_learnable_delay or not isinstance(source_neuron, Input):
            # Both add_to_pre and delay learning is based
            # on the difference of dx_dt and dx_dtplusm
            # synapse equations first:
            dx_dt_diff_sum = sum(
                _get_lmd_sym(syn_sym) * (syn_dx_dt_plus_m[syn_sym] - syn_expr)
                for syn_sym, syn_expr in synapse_model.dx_dt.items())

            # then neuron equations:
            # **NOTE** if this is a delta synapse, nothing is 
            # injected via ISyn apart from at jump-time so we should 
            # be subtracting neuron dynamics with NO synaptic input
            inject_expr = (0 if synapse_model.is_delta_synapse 
                           else synapse_model.inject_current)
            dx_dt_diff_sum += sum(
                _get_lmd_sym(trg_sym) * (trg_dx_dt_plus_m[trg_sym] 
                                        - trg_expr.subs(isyn_sym, inject_expr))
                for trg_sym, trg_expr in trg_neuron_model.dx_dt.items())

            # and simplify
            dx_dt_diff_sum = sympy.simplify(dx_dt_diff_sum)

            # If any target neuron parameters are referenced in 
            # add to pre expression, duplicate in weight update model
            _add_required_wum_post_neuron_parameters(
                trg_neuron_model, genn_model, dx_dt_diff_sum,
                compile_state.optimisers.get(trg_pop, {}))

            # If any synapse parameters are referenced in add to pre
            # expression, duplicate in weight update model
            _add_required_wum_psm_parameters(synapse_model, genn_model,
                                             dx_dt_diff_sum)

            # If any target population variables or lambda variables are 
            # referenced, add neuron variable references and, if connection
            # is delayed, put $ at the start of symbol names in expression 
            # so they can be replaced by the template engine
            for trg_var in trg_neuron_model.dx_dt.keys():
                if dx_dt_diff_sum.has(trg_var):
                    raise NotImplementedError(
                        "Synapse equations which require saving forward "
                        "pass neuron variables are currently not supported.")

                trg_lambda_var = _get_lmd_sym(trg_var)
                if dx_dt_diff_sum.has(trg_lambda_var):
                    genn_model.add_post_neuron_var_ref(trg_lambda_var.name, 
                                                       "scalar", 
                                                       trg_lambda_var.name)
                    if has_delay:
                        dx_dt_diff_sum = _template_symbol(dx_dt_diff_sum,
                                                          trg_lambda_var)

            # If any synapse model variables or lambda variables are 
            # referenced, add psm variable references
            for syn_var in synapse_model.dx_dt.keys():
                if dx_dt_diff_sum.has(syn_var):
                    raise NotImplementedError(
                        "Synapse equations which require saving forward "
                        "pass synapse variables are currently not supported.")

                syn_lambda_var = _get_lmd_sym(syn_var)
                if dx_dt_diff_sum.has(syn_lambda_var):
                    genn_model.add_psm_var_ref(syn_lambda_var.name, 
                                               "scalar", 
                                               syn_lambda_var.name)
                    if has_delay:
                        dx_dt_diff_sum = _template_symbol(dx_dt_diff_sum,
                                                          syn_lambda_var)

            # Use template engine to add delay subscript
            # to any references to lambda variables
            dx_dt_diff_sum_code =\
                Template(sympy.ccode(dx_dt_diff_sum)).substitute(
                    {_get_lmd_name(p): f"{_get_lmd_name(p)}[{int_delay_name}]" 
                     for p in chain(trg_neuron_model.dx_dt.keys(),
                                    synapse_model.dx_dt.keys())})

        #---------------------------------------------------------------------
        # Step 3: Update weight and delay gradients
        #---------------------------------------------------------------------
        # If connection is trainable
        if connect_snippet.trainable:
            # Ensure weights are instantiated as a state variable
            genn_model.make_param_var("weight")

            # Add weight gradient
            genn_model.add_var("weightGradient", "scalar", 0.0)

            # If connection is delayed, add delay index calculation
            if has_learnable_delay:
                genn_model.append_pre_event_syn_code(
                    "const int delayInt = max(0, min(MaxDelay, (int)round(delay)));")
    
            # Assemble gradient update
            weight_grad_update = 0
            for jump_sym, jump_expr in synapse_model.jumps.items():
                lambda_sym = _get_lmd_sym(jump_sym)
                weight_grad_update -= (lambda_sym 
                                       * sympy.diff(jump_expr, 
                                                    sympy.Symbol("weight")))
                if not genn_model.has_psm_var_ref(lambda_sym.name):
                    genn_model.add_psm_var_ref(lambda_sym.name, "scalar", lambda_sym.name)

                # If connection has delays, add $ to name so we 
                # can use template engine to add delay indexing
                if has_delay:
                    weight_grad_update = _template_symbol(weight_grad_update,
                                                          lambda_sym)

            # If this is a delta synapse
            if synapse_model.is_delta_synapse:
                # Loop through neuron variables
                for trg_sym, trg_expr in trg_neuron_model.dx_dt.items():
                    # If input is injected into this variable
                    if trg_expr.has(isyn_sym):
                        lambda_sym = _get_lmd_sym(trg_sym)
                        weight_grad_update -= lambda_sym

                        # Add variable reference if required
                        if not genn_model.has_post_neuron_var_ref(lambda_sym.name):
                            genn_model.add_post_neuron_var_ref(lambda_sym.name, 
                                                               "scalar", 
                                                               lambda_sym.name)
                        
                        # If connection has delays, add $ to name so we 
                        # can use template engine to add delay indexing
                        if has_delay:
                            weight_grad_update = _template_symbol(weight_grad_update,
                                                                  lambda_sym)

            logger.debug(f"\tWeight gradient update: {weight_grad_update}")
            weight_grad_update_code =\
                Template(sympy.ccode(weight_grad_update)).substitute(
                    {_get_lmd_name(p): f"{_get_lmd_name(p)}[{int_delay_name}]" 
                     for p in synapse_model.jumps.keys()})
            genn_model.append_pre_event_syn_code(
                f"weightGradient += {weight_grad_update_code};")
            
            # If any synapse parameters are referenced in gradient 
            # update expression, duplicate in weight update model
            _add_required_wum_psm_parameters(synapse_model, genn_model,
                                             weight_grad_update)
        
            # If delays can be learned
            if has_learnable_delay:
                # Ensure delays are instantiated as a state variable
                genn_model.make_param_var("delay")

                # Add delay gradient
                genn_model.add_var("delayGradient", "scalar", 0.0)

                # Add delay calculation
                logger.debug(f"\tDelay gradient update: {dx_dt_diff_sum}")
                genn_model.append_pre_event_syn_code(
                    f"delayGradient += {dx_dt_diff_sum_code};")

        #---------------------------------------------------------------------
        # Step 4: Backpropagate gradients
        #---------------------------------------------------------------------
        # If source neuron isn't an input neuron
        if not isinstance(source_neuron, Input):
            # There should be some gradient injection!
            assert dx_dt_diff_sum != 0

            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)
            logger.debug(f"\tAdd to pre: {dx_dt_diff_sum}")

            # Convert expression to c-code and insert call to addToPre
            genn_model.append_pre_event_syn_code(f"addToPre({dx_dt_diff_sum_code});")

        return genn_model

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict, 
                                compile_state: CompileState) -> CompiledTrainingNetwork:
        # Correctly target feedback
        for c in compile_state.feedback_connections:
            connection_populations[c].pre_target_var = "RevISyn"

        # Loop through connections and populations that require optimisers
        optimisers = []
        checkpoint_connection_vars = []
        checkpoint_population_vars = []
        need_zero_gradient_update_group = False
        i = 0
        for k, vars in compile_state.optimisers.items():
            # If key is a connection
            if k in connection_populations:
                # If weight optimisation is required
                genn_pop = connection_populations[k]
                gradient_vars = []
                if "weight" in vars:
                    # Create weight optimiser custom update
                    cu_weight = self._create_optimiser_custom_update(
                        f"Weight{i}", create_wu_var_ref(genn_pop, "weight"),
                        create_wu_var_ref(genn_pop, "weightGradient"), 
                        vars["weight"], genn_model)
            
                    # Add custom update to list of optimisers
                    optimisers.append((vars["weight"], cu_weight))
                    
                    # Add variable to list of those to checkpoint
                    checkpoint_connection_vars.append((k, "weight"))
                    
                    # Add gradient to list of gradient vars to zero
                    gradient_vars.append(("weightGradient", "scalar", 0.0))
                
                # If delay optimiser is required
                if "delay" in vars:
                    # Create delay optimiser custom update
                    cu_delay = self._create_optimiser_custom_update(
                        f"Delay{i}", create_wu_var_ref(genn_pop, "delay"),
                        create_wu_var_ref(genn_pop, "delayGradient"),
                        vars["delay"], genn_model,
                        (0.0, k.max_delay_steps))

                    # Add custom update to list of optimisers
                    optimisers.append((vars["delay"], cu_delay))

                    # Add variable to list of those to checkpoint
                    checkpoint_connection_vars.append((k, "delay"))

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
                need_zero_gradient_update_group = True
                i += 1
            # Otherwise if key is a population
            elif k in neuron_populations:
                genn_pop = neuron_populations[k]
                for n, o in vars.items():
                    # Create parameter optimiser custom update
                    cu_param = self._create_optimiser_custom_update(
                        f"{n}{i}", create_var_ref(genn_pop, n),
                            create_var_ref(genn_pop, f"{n}Gradient"),
                            o, genn_model)

                    # Add custom update to list of optimisers
                    optimisers.append((o, cu_param))
                    checkpoint_population_vars.append((k, n))

                i += 1
            else:
                assert False

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

        # Add per-batch adjoint limit custom updates for each population and adjoint var that requires them
        for p, var in compile_state.adjoint_limit_pops_vars:
            genn_pop = neuron_populations[p]
            self._add_abs_sum_reduce_custom_update(genn_model, genn_pop, var)
        
        # Create bespoke custom updates to handle reduction for
        # relative mean-squared error loss
        for i, p in enumerate(compile_state.ttfs_reduce_populations):
            genn_pop = neuron_populations[p]
            self._create_ttfs_reduce_custom_update(
                genn_model, genn_pop, self.dt * self.example_timesteps,
                f"CUTTFSReduce{i}")

        # Build list of base callbacks
        base_train_callbacks = []
        base_validate_callbacks = []
        if len(optimisers) > 0:
            if self.full_batch_size > 1:
                base_train_callbacks.append(
                    CustomUpdateOnBatchEndNotFirst("GradientBatchReduce"))
            base_train_callbacks.append(
                CustomUpdateOnBatchEndNotFirst("GradientLearn"))
            if need_zero_gradient_update_group:
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
        
        # Add custom uopdate for adjoint limit calculation if required
        if len(compile_state.adjoint_limit_pops_vars) > 0:
            base_train_callbacks.append(CustomUpdateOnBatchEndNotFirst("AbsSumReduceBatch"))
            base_train_callbacks.append(CustomUpdateOnBatchEndNotFirst("ReduceAssign"))

        # If spike count reduction is required at end of batch, add callback
        if len(compile_state.spike_count_populations) > 0 and self.full_batch_size > 1:
            base_train_callbacks.append(CustomUpdateOnBatchEnd("SpikeCountReduce"))

        # If TTFS needs reducing, add reduction callback before reset
        if len(compile_state.ttfs_reduce_populations) > 0:
            base_train_callbacks.append(CustomUpdateOnBatchBegin("TTFSReduce"))

        # Add callbacks for loss recording
        for pop, key in compile_state.loss_recorder_populations:
            base_train_callbacks.append(LossRecorderCallback(pop, key))
            base_validate_callbacks.append(LossRecorderCallback(pop, key))

        # Add reset custom updates
        if compile_state.is_reset_custom_update_required:
            base_train_callbacks.append(CustomUpdateOnBatchBegin("Reset"))
            base_validate_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations,
            self.communicator, compile_state.ground_truths,
            self.example_timesteps, base_train_callbacks,
            base_validate_callbacks, optimisers,
            checkpoint_connection_vars, checkpoint_population_vars, True)

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

    def _add_abs_sum_reduce_custom_update(self, genn_model, genn_pop, var):
        reduce_batch = CustomUpdateModel(
            abs_sum_reduce_batch_model, {}, {"BRedAbsSum": 0.0},
            {"AbsSum": create_var_ref(genn_pop, f"{var}AbsSum")})
        genn_reduce_batch = self.add_custom_update(
            genn_model, reduce_batch, "AbsSumReduceBatch", 
            "AbsSumReduceBatch" + genn_pop.name + var)

        reduce_assign = CustomUpdateModel(
            abs_sum_reduce_neuron_model_assign, 
            {"timesteps": self.example_timesteps,"gradLimit": self.grad_limit}, {},
            {"BRedAbsSum": create_var_ref(genn_reduce_batch, "BRedAbsSum"),
             "Limit": create_var_ref(genn_pop, f"{var}Limit")})
        self.add_custom_update(genn_model, reduce_assign, "ReduceAssign",
                               "ReduceAssign" + genn_pop.name + var)

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
            if isinstance(var_ref, WUVarReference):
                reduced_gradient = create_wu_var_ref(genn_reduction,
                                                     "ReducedGradient")
            else:
                reduced_gradient = create_var_ref(genn_reduction,
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
            self.add_custom_update(genn_model, reduction_optimiser_model,
                                   "SpikeCountReduce", name)

    def _create_ttfs_reduce_custom_update(self, genn_model,
                                          genn_pop, example_time: float,
                                          name: str):
        # Create model which sums valid first spike times into TFirstSpikeSumBack
        # and selects first spike time from true output
        reduce_model = CustomUpdateModel(
            model={"var_refs": [("YTrue", "uint8_t", VarAccessMode.READ_ONLY),
                                ("TFirstSpike", "scalar", VarAccessMode.READ_ONLY),
                                ("TFirstSpikeSumBack", "scalar", VarAccessMode.REDUCE_SUM),
                                ("TFirstSpikeTrueBack", "scalar", VarAccessMode.REDUCE_SUM)],
                   "update_code": f"""
                       TFirstSpikeTrueBack = (id == YTrue && TFirstSpike >= -{example_time}) ? TFirstSpike : 0.0;
                       TFirstSpikeSumBack = (TFirstSpike >= -{example_time}) ? TFirstSpike : 0.0;
                       """},
            var_refs={"YTrue": create_var_ref(genn_pop, "YTrue"),
                      "TFirstSpike": create_var_ref(genn_pop, "TFirstSpike"),
                      "TFirstSpikeSumBack": create_var_ref(genn_pop, "TFirstSpikeSumBack"),
                      "TFirstSpikeTrueBack": create_var_ref(genn_pop, "TFirstSpikeTrueBack")})

        # Add GeNN custom update to model
        self.add_custom_update(genn_model, reduce_model, "TTFSReduce", name)

    def _build_adjoint_system(self, model: AutoNeuronModel, learn_params,
                              output: bool, regularise: bool):
        logger.debug("\tBuilding adjoint system for AutoNeuronModel:")
        logger.debug(f"\t\tVariables: {model.var_vals.keys()}")
        logger.debug(f"\t\tParameters: {model.param_vals.keys()}")
        logger.debug(f"\t\tForward ODEs: {model.dx_dt}")
        logger.debug(f"\t\tForward jumps: {model.jumps}")

        # generate adjoint ODE
        # assume that neuron variables do not appear in rhs of post-synapse ODEs
        # therefore, we can do these independent of synapse equations
        saved_vars_timestep = set()
        saved_vars_spike = set()
        dl_dt = {}
        for sym in model.dx_dt.keys():
            o = sum(sympy.diff(expr2, sym) * _get_lmd_sym(sym2)
                    for sym2, expr2 in model.dx_dt.items())
            
            # collect variables they might need to go into a ring buffer:
            o = _template_symbols(
                o, chain(model.var_vals.keys(), ["Isyn"]), saved_vars_timestep)
            dl_dt[_get_lmd_sym(sym)] = o

        logger.debug(f"\t\tAdjoint ODE: {dl_dt}")

        # create expressions for learned neuron parameters
        grad_terms = {}
        for p in learn_params:
            sym = sympy.Symbol(p)
            o = - sum(sympy.diff(expr2, sym) * _get_lmd_sym(sym2)
                    for sym2, expr2 in model.dx_dt.items())
            # collect variables they might need to go into a ring buffer:
            o = _template_symbols(
                o, chain(model.var_vals.keys(), ["Isyn"]), saved_vars_timestep)
            grad_terms[sym] = o

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
            saved_vars_spike.update({var_name2 for var_name2 in model.var_vals.keys()
                                     if a[var_sym].has(sympy.Symbol(var_name2))})

            if model.threshold != 0:
                ex = sympy.simplify(
                    sum(sympy.diff(model.threshold, var_sym2) * var_expr2
                        for var_sym2, var_expr2 in model.dx_dt.items()))
                if ex != 0:
                    ex = sympy.diff(model.threshold, var_sym) / ex
                    if ex != 0:
                        b[var_sym] = _simplify_using_threshold(
                            model.var_vals.keys(), model.threshold, ex)

                        # **TODO** helper
                        saved_vars_spike.update(
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
                            model.var_vals.keys(), model.threshold, ex)
                        saved_vars_spike.update(
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
                # add l^- - l^+ jump for neurons with regularisation
                if regularise:
                    # scaling factor is made so that jumps lead to an area of size 1
                    # to be added to the integral of the "invisible trace variable"
                    # underlying the regularisation loss
                    drive += sympy.Symbol("drive_reg")/(sympy.Symbol("t")+self.dt)
                jump = a_exp + b[a_sym] * (ex2 + drive)
            else:
                jump = a_exp
            jump =  _simplify_using_threshold(model.var_vals.keys(),
                                              model.threshold, jump)
            
            # If any jumping remains
            if jump != 0:
                # Template any neuron variable names or Isyn
                jump = _template_symbols(
                    jump, chain(model.var_vals.keys(), ["Isyn"]), saved_vars_spike)
                
                # Add to dictionary as jump for adjoint variable
                adjoint_jumps[_get_lmd_sym(a_sym)] = jump

        logger.debug(f"\t\tAdjoint Jumps: {adjoint_jumps}")
        logger.debug(f"\t\tSaved variables per timestep: {saved_vars_timestep}")
        logger.debug(f"\t\tSaved variables per spike: {saved_vars_spike}")
        
        return dl_dt, adjoint_jumps, grad_terms, saved_vars_timestep, saved_vars_spike
    
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

        # add "Trial" global parameter and add the population to list of those who need it
        genn_model.add_param("Trial", "unsigned int", 0)
        genn_model.set_param_dynamic("Trial")
        compile_state.update_trial_pops.append(pop)

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
            write_code_timestep = ""
            write_code = ""
            tsringoffset = ""
        # Otherwise i.e. it's hidden
        else:
            logger.debug(f"Building hidden neuron model for '{pop.name}'")
            if not isinstance(model, AutoNeuronModel):
                raise NotImplementedError(
                    "EventProp compiler only supports hidden "
                    "neurons defined in terms of AutoNeuronModel")
            
            # Build adjoint system from model
            learn_params = compile_state.optimisers.get(pop, {})
            regularise = (self.reg_lambda_lower != 0.0) or (self.reg_lambda_upper != 0.0)
            dl_dt, adjoint_jumps, grad_terms, saved_vars_timestep, saved_vars_spike =\
                self._build_adjoint_system(model, learn_params, False, regularise)

            additional_reset_vars = []
            # Add variables for parameter gradients
            for p in learn_params:
                genn_model.add_var(f"{p}Gradient", "scalar", 0.0)
                genn_model.make_param_var(p)

            # Add EGPs for adjoint variable value limits (to avoid pathological large jumps)
            dynamics_code = ""
            for jump_sym in adjoint_jumps.keys():
                genn_model.add_var(f"{jump_sym.name}AbsSum", "scalar", 0.0) 
                genn_model.add_var(f"{jump_sym.name}Limit", "scalar", 1.0,
                                   VarAccess.READ_ONLY_SHARED_NEURON, False)
                compile_state.adjoint_limit_pops_vars.append((pop, jump_sym.name))
                dynamics_code += f"{jump_sym.name}AbsSum += fabs({jump_sym.name});\n"

            # Generate transition code
            transition_code = "\n".join(
                f"{jump_sym.name} = max(min({sympy.ccode(jump_expr)},{jump_sym.name}Limit),-{jump_sym.name}Limit);"
                for jump_sym, jump_expr in adjoint_jumps.items())
            
            # Substitute saved variables for those in the ring buffer
            transition_code = Template(transition_code).substitute(
                {s: f"Ring{s}[ringOffset + RingReadOffset]" for s in saved_vars_spike})

            # Add additional input variable to receive add_to_pre feedback
            genn_model.add_additional_input_var("RevISyn", "scalar", 0.0)

            # Add EGP for stored vars ring variables
            dyn_ts_reset_needed = False
            timestep_ring_size = self.batch_size * np.prod(pop.shape) * 2 * self.example_timesteps
            for var in saved_vars_timestep:
                genn_model.add_egp(f"tsRing{var}", "scalar*", 
                                   np.empty(timestep_ring_size, dtype=np.float32))
                dyn_ts_reset_needed = True
            spike_ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            for var in saved_vars_spike:
                genn_model.add_egp(f"Ring{var}", "scalar*", 
                                   np.empty(spike_ring_size, dtype=np.float32))

            # Add state variables and reset for lambda variables
            for lambda_sym in dl_dt.keys():
                genn_model.add_var(lambda_sym.name, "scalar", 0.0)

            # If regularisation is enabled
            # **THINK** is this LIF-specific?
            if regularise and model.threshold != 0:
                logger.debug("\tBuilding regulariser")
                # Add state variables to hold spike count
                # during forward and backward pass. 
                # **NOTE** SpikeCountBackSum is shared across
                # batches as it is the result of a reduction
                # **NOTE** if batch size > 1, SpikeCountBackBatch is
                # calculated with a reduction
                # **NOTE** SpikeCount was previously zeroed in the reduction
                # meaning it wasn't getting reset after validation examples
                genn_model.add_var("SpikeCount", "int", 0, reset=True)
                genn_model.add_var("SpikeCountBackBatch", "int", 0,
                                   VarAccess.READ_ONLY, reset=False)

                # Add parameters for regulariser
                # **NOTE** this is multiplied by batch_size so it
                # can be compared directly to SpikeCountBackBatch
                genn_model.add_param("RegNuUpperBatch", "int",
                                     self.reg_nu_upper * self.full_batch_size)
                # If batch size is 1, add reset variables to copy SpikeCount
                # into SpikeCountBackBatch and zero SpikeCount
                if self.full_batch_size == 1:
                    additional_reset_vars.append(
                        ("SpikeCountBackBatch", "int", "SpikeCount"))

                # Calculate regularisation drive
                # We divide by batch size by formulation of the loss function and then again to take into consideration that
                # SpikeCountBackBatch is collected across a batch; but then we multiply by reg_nu_upper times batch size
                # to normalise the effect of dividing by SpikeCountBackBatch.
                # The division by SpikeCountBackBatch is motivated by the observation that the drive_reg is applied
                # number of spike times, which biases regularisation towards suppressing too many spikes over enhancing to few
                dynamics_code += f"""
                scalar drive_reg;
                const scalar spikeDev = (SpikeCountBackBatch - RegNuUpperBatch);
                if (spikeDev > 0.0) {{
                    drive_reg = -{self.reg_lambda_upper/self.full_batch_size/self.full_batch_size} * spikeDev;
                }}
                else {{
                    drive_reg = -{self.reg_lambda_lower/self.full_batch_size/self.full_batch_size} * spikeDev;
                }}
                """

                # Add population to list of those that 
                # require a spike count reduction
                compile_state.spike_count_populations.append(pop)

                # Add code to update SpikeCount in forward reset code
                genn_model.append_reset_code("SpikeCount++;")

            # Add reset logic to reset state variables (including adjoint)
            # **NOTE** additional reset vars first so reset happens in correct order
            all_reset_vars = additional_reset_vars + genn_model.reset_vars
            logger.debug(f"\tReset variables: {all_reset_vars}")
            compile_state.add_neuron_reset_vars(
                pop, all_reset_vars, True, dyn_ts_reset_needed)

            # Generate ring-buffer write code
            write_code_timestep = "\n".join(f"tsRing{v}[tsRingOffset + tsRingWriteOffset] = {v};"
                                            for v in saved_vars_timestep)

            if dyn_ts_reset_needed:
                write_code_timestep += f"""
                tsRingWriteOffset++;
                """
                # add read and write pointer for timestep-wise ring buffers
                genn_model.add_var("tsRingWriteOffset", "int", 0, reset=False)
                genn_model.add_var("tsRingReadOffset", "int", self.example_timesteps, reset=False)
                read_pointer_code= "tsRingReadOffset--;"
                tsringoffset = f"const int tsRingOffset = (batch * num_neurons * {self.example_timesteps * 2}) + (id * {self.example_timesteps} * 2);"

            else:
                read_pointer_code= ""
                tsringoffset = ""

            write_code ="\n".join(f"Ring{v}[ringOffset + RingWriteOffset] = {v};"
                                  for v in saved_vars_spike)

            # Solve ODE and generate dynamics code
            dynamics_code += solve_ode(dl_dt, model.solver, model.sub_steps)
            # Add gradient accumulation code
            for p, expr in grad_terms.items():
                dynamics_code += f"{p}Gradient += {sympy.ccode(expr)};\n"
            dynamics_code = Template(dynamics_code).substitute(
                {s: f"tsRing{s}[tsRingOffset + tsRingReadOffset]" for s in saved_vars_timestep})

            dynamics_code = f"""
                {read_pointer_code}
                {dynamics_code}
            """

        # Add code to start of sim code to run 
        # backwards pass and handle back spikes
        genn_model.prepend_sim_code(
            neuron_backward_pass.substitute(
                max_spikes=self.max_spikes,
                example_time=(self.example_timesteps * self.dt),
                dynamics=dynamics_code,
                transition=transition_code,
                example_timesteps=self.example_timesteps,
                write=write_code_timestep,
                tsringoffset=tsringoffset
            ))

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
        
        # Add forward and backward-pass ground truth state to model
        ground_truth = compile_state.ground_truths[pop]
        ground_truth.add_to_neuron(True, genn_model, pop.shape,
                                   self.batch_size, self.example_timesteps)

        # Add output logic to model
        pop.neuron.readout.add_readout_logic(
            genn_model, max_time_required=True, dt=self.dt,
            example_timesteps=self.example_timesteps)

        # Build adjoint system from model
        learn_params = compile_state.optimisers.get(pop, {})
        dl_dt, adjoint_jumps, grad_terms, saved_vars_timestep, saved_vars_spike =\
            self._build_adjoint_system(model, learn_params, True, False)

        # Add variables for parameter gradients
        for p in learn_params:
            genn_model.add_var(f"{p}Gradient", "scalar", 0.0)

        # Add continous drive term to LambdaV
        dl_dt[_get_lmd_sym(model.output_var_name)] += sympy.Symbol("drive")

        # Add adjoint state variables
        # **THINK** what about reset
        for lambda_sym in dl_dt.keys():
            genn_model.add_var(lambda_sym.name, "scalar", 0.0)

        # Add EGP for stored vars ring variables
        dyn_ts_reset_needed = False
        timestep_ring_size = self.batch_size * np.prod(pop.shape) * 2 * self.example_timesteps
        for var in saved_vars_timestep:
            genn_model.add_egp(f"tsRing{var}", "scalar*", 
                               np.empty(timestep_ring_size, dtype=np.float32))
            dyn_ts_reset_needed = True

        # add read and write pointer for timestep-wise ring buffers
        if dyn_ts_reset_needed:
            genn_model.add_var("tsRingWriteOffset", "int", 0, reset=False)
            genn_model.add_var("tsRingReadOffset", "int", self.example_timesteps, reset=False)
            read_pointer_code= "tsRingReadOffset--;"
            write_pointer_code= "tsRingWriteOffset++;"
            tsringoffset = f"const int tsRingOffset = (batch * num_neurons * {self.example_timesteps * 2}) + (id * {self.example_timesteps} * 2);"
        else:
            read_pointer_code= ""
            write_pointer_code= ""
            tsringoffset = ""
            
        # Generate ring-buffer write code
        write_code_timestep = "\n".join(f"tsRing{v}[tsRingOffset + tsRingWriteOffset] = {v};"
                                        for v in saved_vars_timestep)

        # Prepend continous adjoint system update
        dynamics_code = solve_ode(dl_dt, model.solver, model.sub_steps)
        # Add gradient accumulation code
        for p_grad, expr in grad_terms.items():
            dynamics_code += f"{p_grad} += {sympy.ccode(expr)};\n"
        dynamics_code = Template(dynamics_code).substitute(
            {s: f"tsRing{s}[tsRingOffset + tsRingReadOffset]" for s in saved_vars_timestep})

        # Add dynamic parameter to contain trial index and add 
        # population to list of those which require it updating
        genn_model.add_param("Trial", "unsigned int", 0)
        genn_model.set_param_dynamic("Trial")
        compile_state.update_trial_pops.append(pop)

        # Variables to track what affect loss function has on reset logic
        additional_reset_vars = ground_truth.backward_duplicate_var_reset
        reset_event_ring = False
        reset_v_ring = dyn_ts_reset_needed
        
        # If we should record loss
        pop_loss = compile_state.losses[pop]
        if pop_loss.record_key is not None:
            # Add variable to record into
            genn_model.add_var("LossSum", "scalar", 0.0)

            # Define code generator function
            gen_record_code = lambda n: f"LossSum += -log({n});"
            
            # Add population to list 
            compile_state.loss_recorder_populations.append(
                (pop, pop_loss.record_key))
        # Otherwise don't generate any code
        else:
            gen_record_code = lambda n: ""
            
        # If model is non-spiking - MSE and SCE losses of "voltage V" apply
        sce_loss = isinstance(pop_loss, SparseCategoricalCrossentropy)
        mse_loss = isinstance(pop_loss, MeanSquareError)
        per_neuron_mse_loss = isinstance(pop_loss, PerNeuronMeanSquareError)
        rmse_loss = isinstance(pop_loss, RelativeMeanSquareError)
        if "threshold" not in model.model or model.model["threshold"] is None:
            # Check adjoint system is also jump-less
            assert len(saved_vars_spike) == 0
            assert len(adjoint_jumps) == 0

            # If we want to calculate mean-squared error or per-timestep loss
            out_var_name = model.output_var_name
            if self.per_timestep_loss or mse_loss:
                # Add variables to hold offsets for 
                # reading and writing ring variables if not yet done for dynamics
                if not dyn_ts_reset_needed:
                    genn_model.add_var("tsRingWriteOffset", "int", 0, reset=False)
                    genn_model.add_var("tsRingReadOffset", "int", 0, reset=False)
 
                # Add EGP for softmax V (SCE) or regression difference (MSE) ring variable
                ring_size = self.batch_size * np.prod(pop.shape) * 2 * self.example_timesteps
                genn_model.add_egp("RingOutputLossTerm", "scalar*", 
                                   np.empty(ring_size, dtype=np.float32))

                # We have a variable ring so it will need resetting
                reset_v_ring = True

                # If sparse cross-entropy loss is selected
                if sce_loss:
                    # If readout is AvgVar or SumVar
                    if isinstance(pop.neuron.readout, (AvgVar, SumVar)):
                        genn_model.prepend_sim_code(
                            f"""
                            scalar drive = 0.0;
                            const int tsRingOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                            if (Trial > 0) {{
                                tsRingReadOffset--;
                                const scalar loss = RingOutputLossTerm[tsRingOffset + tsRingReadOffset];

                                if(id == YTrueBack) {{
                                    {gen_record_code('loss')}
                                    drive = (1.0 - loss) / (num_batch * {self.dt * self.example_timesteps});
                                }}
                                else {{
                                    drive = -loss / (num_batch * {self.dt * self.example_timesteps});
                                }}
                            }}

                            {dynamics_code}
                            """)

                        # Add custom updates to calculate 
                        # softmax from V and write directly to buffermodel_copy
                        compile_state.timestep_softmax_populations.append(
                            (pop, out_var_name))
                    # Otherwise, unsupported readout type
                    else:
                        raise NotImplementedError(
                            f"EventProp compiler with CategoricalCrossEntropy loss doesn't support "
                            f"{type(pop.neuron.readout).__name__} readouts")
                elif mse_loss:
                    assert isinstance(pop.neuron.readout, Var)
                    genn_model.prepend_sim_code(
                        f"""
                        scalar drive = 0.0;
                        const int tsRingOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                        if (Trial > 0) {{
                            tsRingReadOffset--;
                            const scalar loss = RingOutputLossTerm[tsRingOffset + tsRingReadOffset];
                            {gen_record_code('loss')}
                            drive = loss / (num_batch * {self.dt * self.example_timesteps});
                        }}
                        
                        {dynamics_code}
                        """)

                    # Add code to fill errors into RingBuffer
                    genn_model.append_sim_code(
                        f"""
                        const unsigned int timestep = (int)round(t / dt);
                        const unsigned int index = (batch * {self.example_timesteps} * num_neurons) + (timestep * num_neurons) + id;
                        RingOutputLossTerm[tsRingOffset + tsRingWriteOffset] = YTrue[index] - {out_var_name};
                        tsRingWriteOffset++;
                        """) 
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
                            scalar drive = 0.0;
                            if (Trial > 0) {{
                                if(id == YTrueBack) {{
                                    {gen_record_code('Softmax')}
                                    drive = (1.0 - Softmax) / (num_batch * {self.dt * self.example_timesteps});
                                }}
                                else {{
                                    drive = -Softmax / (num_batch * {self.dt * self.example_timesteps});
                                }}
                            }}
                            {read_pointer_code}
                            {dynamics_code}
                            {write_pointer_code}
                            """)
                    
                        # Add custom updates to calculate 
                        # softmax from VSum or VAvg
                        suffix = ("Sum" if isinstance(pop.neuron.readout, SumVar)
                                  else "Avg")
                        compile_state.batch_softmax_populations.append(
                            (pop, out_var_name + suffix, "Softmax"))
                    # Otherwise, if genn_model is AvgVarExpWeight
                    elif isinstance(pop.neuron.readout, AvgVarExpWeight):
                        local_t_scale = 1.0 / (self.dt * self.example_timesteps)
                        genn_model.prepend_sim_code(
                            f"""
                            // Backward pass
                            scalar drive = 0.0;
                            if (Trial > 0) {{
                                if(id == YTrueBack) {{
                                    {gen_record_code('Softmax')}
                                    drive = ((1.0 - Softmax) * exp(-(1.0 - (t * {local_t_scale})))) / (num_batch * {self.dt * self.example_timesteps});
                                }}
                                else {{
                                    drive = -(-Softmax * exp(-(1.0 - (t * {local_t_scale})))) / (num_batch * {self.dt * self.example_timesteps});
                                }}
                            }}
                            {read_pointer_code}
                            {dynamics_code}
                            {write_pointer_code}
                            """)
                    
                        # Add custom updates to calculate softmax from VAvg
                        compile_state.batch_softmax_populations.append(
                            (pop, out_var_name + "Avg", "Softmax"))

                    # Otherwise, if readout is MaxVar
                    elif isinstance(pop.neuron.readout, MaxVar):
                        # Add state variable to hold vmax from previous trial
                        genn_model.add_var(model.output_var_name + "MaxTimeBack", "scalar", 0.0,
                                           VarAccess.READ_ONLY_DUPLICATE, reset=False)

                        genn_model.prepend_sim_code(
                            f"""
                            const scalar backT = {self.example_timesteps * self.dt} - t - dt;
                            scalar drive = 0.0;
                            if (Trial > 0 && fabs(backT - {out_var_name}MaxTimeBack) < 1e-3*dt) {{
                                if(id == YTrueBack) {{
                                    {gen_record_code('Softmax')}
                                    drive = (1.0 - Softmax) / (num_batch * {self.dt * self.example_timesteps});
                                }}
                                else {{
                                    drive = -Softmax / (num_batch * {self.dt * self.example_timesteps});
                                }}
                            }}
                            {read_pointer_code}
                            {dynamics_code}
                            {write_pointer_code}
                            """)
                    
                        # Add custom updates to calculate softmax from VMax
                        compile_state.batch_softmax_populations.append(
                            (pop, model.output_var_name + "Max", "Softmax"))

                        # Additionally reset VMaxTimeBack to VMaxTime
                        # **NOTE** reset VMaxTimeBack first so VMaxTime isn't zeroed
                        # **TODO** time type
                        additional_reset_vars.append(
                            (f"{out_var_name}MaxTimeBack", "scalar",
                             f"{out_var_name}MaxTime"))
                    elif isinstance(pop.neuron.readout, EndVar):
                        genn_model.prepend_sim_code(
                            f"""
                            scalar drive = 0.0;
                            if (Trial > 0 && t < 1e-3*dt) {{
                                const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                                drive = g / (num_batch * {self.dt * self.example_timesteps});
                            }}
                            {read_pointer_code}
                            {dynamics_code}
                            {write_pointer_code}
                            """)

                        # Add custom updates to calculate softmax from output variable
                        compile_state.batch_softmax_populations.append(
                            (pop, model.output_var_name, "Softmax"))
                    # Otherwise, unsupported readout type
                    else:
                        raise NotImplementedError(
                            f"EventProp compiler doesn't support "
                            f"{type(pop.neuron.readout).__name__} readouts")
                elif mse_loss:
                    raise NotImplementedError(
                        f"EventProp compiler doesn't support "
                        f"time averaged loss for regression.")
        # Otherwise, output neuron is spiking
        else:
            # Generate transition code
            transition_code = "\n".join(
                f"{sympy.ccode(jump_expr, assign_to=jump_sym.name)};"
                for jump_sym, jump_expr in adjoint_jumps.items())
            
            # Substitute saved variables for those in the ring buffer
            transition_code = Template(transition_code).substitute(
                {s: f"Ring{s}[ringOffset + RingReadOffset]" for s in saved_vars_spike})

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
            
            # We always need to reset the event ring for these losses
            reset_event_ring = True

            # Add EGP for spike time ring variables
            spike_ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            genn_model.add_egp("RingSpikeTime", "scalar*", 
                                np.empty(spike_ring_size, dtype=np.float32))
            
            # Add EGP for stored vars ring variables
            spike_ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            for var in saved_vars_spike:
                genn_model.add_egp(f"Ring{var}", "scalar*", 
                                   np.empty(spike_ring_size, dtype=np.float32))

            # Add parameters with synaptic decay and scale constants
            #model_copy.add_param("IsynScale", "scalar",
            #    self.dt / (tau_syn  * (1.0 - beta)))

            # Readout has to be FirstSpikeTime
            if isinstance(pop.neuron.readout, FirstSpikeTime):
                # Add state variable to hold TFirstSpike from previous trial
                # **YUCK** REALLY should be timepoint but then you can't softmax
                genn_model.add_var("TFirstSpikeBack", "scalar", 0.0,
                                   VarAccess.READ_ONLY_DUPLICATE, reset=False)

                # Reset TFirstSpikeBack to TFirstSpike
                additional_reset_vars.append(
                    ("TFirstSpikeBack", "scalar", "TFirstSpike"))

                example_time = self.dt * self.example_timesteps
                if sce_loss:
                    # Add state variable to hold softmax of output
                    genn_model.add_var("Softmax", "scalar", 0.0,
                                       VarAccess.READ_ONLY_DUPLICATE, reset=False)

                    # On backward pass transition, update LambdaV if this is the first spike
                    # **THOMAS** why are we dividing by what looks like softmax temperature?
                    # **TODO** build transition_code from adjoint_jumps here
                    transition_code = f"""
                        scalar drive_p = 0.0;
                        if (fabs(backT + TFirstSpikeBack) < 1e-3*dt) {{
                            if (id == YTrueBack) {{
                                const scalar fst = {1.01 * example_time} + TFirstSpikeBack;
                                drive_p = (((1.0 - Softmax) / {self.softmax_temperature}) + ({self.ttfs_alpha} / (fst * fst))) / {self.batch_size};
                                {gen_record_code('Softmax')}
                            }}
                            else {{
                                drive_p = - Softmax / ({self.softmax_temperature * self.batch_size});
                            }}
                        }}
                        {transition_code}
                        """

                    # Add custom updates to calculate softmax from TFirstSpike
                    compile_state.batch_softmax_populations.append(
                        (pop, "TFirstSpike", "Softmax"))
                elif per_neuron_mse_loss:
                    transition_code = f"""
                        scalar drive_p = 0.0;
                        if (fabs(backT + TFirstSpikeBack) < 1e-3*dt) {{
                            drive_p = (-TFirstSpikeBack-YTrueBack);
                            {gen_record_code('drive_p')}
                        }}
                        {transition_code}
                        """
                elif rmse_loss:
                    # Add parameters to model to hold delta
                    genn_model.add_param("Delta", "scalar", 
                                         compile_state.losses[pop].delta)
                    
                    # Add state variable to hold sum of TFirstSpike from previous trial
                    # **YUCK** REALLY should be timepoint but then you can't softmax
                    genn_model.add_var("TFirstSpikeSumBack", "scalar", 0.0,
                                       VarAccess.READ_ONLY_SHARED_NEURON, reset=False)
                    
                    # Add another variable to hold TFirstSpike for the correct 
                    # output neuron from the previous trial at which
                    # **YUCK** REALLY should be timepoint but then you can't softmax
                    genn_model.add_var("TFirstSpikeTrueBack", "scalar", 0.0, 
                                       VarAccess.READ_ONLY_SHARED_NEURON, reset=False)

                    # On backward pass transition, update LambdaV if this is the first spike
                    # **NOTE** Strange term in id == YTrueBack case comes from re-arranging
                    # -(TFirstSpikeBack[n] - TFirstSpikeTrueBack - delta),
                    # summed over all output neurons n and negating spike
                    # times due to design of FirstSpikeTime readout
                    transition_code = f"""
                        scalar drive_p = 0.0;
                        if (fabs(backT + TFirstSpikeBack) < 1e-3*dt) {{
                            if(id == YTrueBack) {{
                                drive_p = (TFirstSpikeSumBack - (num_neurons * TFirstSpikeTrueBack) + ((num_neurons - 1) * Delta));
                                {gen_record_code('drive_p')}
                            }}
                            else {{
                                drive_p = ((-TFirstSpikeBack + TFirstSpikeTrueBack) - Delta);
                            }}
                        }}
                        {transition_code}
                        """

                    # Add population to list requiring bespoke TTFS reduction
                    compile_state.ttfs_reduce_populations.append(pop)

                # Add code to start of sim code to run backwards pass 
                # and handle back spikes with correct LIF dynamics
                genn_model.prepend_sim_code(
                    neuron_backward_pass.substitute(
                        max_spikes=self.max_spikes,
                        example_time=example_time,
                        dynamics=f"""
                            const scalar drive = 0.0;
                            {read_pointer_code}
                            {dynamics_code}
                            {write_pointer_code}
                             """,
                        transition=transition_code,
                        example_timesteps=self.example_timesteps,
                        write=write_code_timestep,
                        tsringoffset=tsringoffset
                    ))

                # Generate ring-buffer write code
                write_code ="\n".join(f"Ring{v}[ringOffset + RingWriteOffset] = {v};"
                                      for v in saved_vars_spike)

                # Prepend (as it accesses the pre-reset value of V) 
                # code to reset to write spike time and saved vars to ring buffer
                reset_code = neuron_reset.substitute(
                    max_spikes=self.max_spikes,
                    write=write_code,
                    strict_check=(neuron_reset_strict_check 
                                  if self.strict_buffer_checking
                                  else ""))
                genn_model.prepend_reset_code(reset_code)

                # Generate 'phantom' spike times
                if sce_loss:
                    # If it's last timestep, neuron hasn't spiked, 
                    # isn't just about to and is correct output, update 
                    # TFirstSpike and insert event into ring-buffer
                    # **THINK** fmax totally uneccessary?
                    genn_model.append_sim_code(
                        f"""
                        if(fabs(t - {example_time - self.dt}) < 1e-3*dt && TFirstSpike < {-example_time} && ({model.model['threshold']}) < 0 && id == YTrue) {{
                            TFirstSpike = fmax(-t, TFirstSpike);
                            {reset_code}
                        }}
                        """)

                elif per_neuron_mse_loss:
                    # If it's last timestep, neuron hasn't spiked, 
                    # isn't just about to and SHOULD spike in this trial,
                    # update TFirstSpike and insert event into ring-buffer
                    # **THINK** fmax totally uneccessary?
                    genn_model.append_sim_code(
                        f"""
                        if(fabs(t - {example_time - self.dt}) < 1e-3*dt && TFirstSpike < {-example_time} && ({model.model['threshold']}) < 0 && YTrue < {example_time}) {{
                            TFirstSpike = fmax(-t, TFirstSpike);
                            {reset_code}
                        }}
                        """)

            # Otherwise, unsupported readout type
            else:
                raise NotImplementedError(
                    f"EventProp compiler with spiking output "
                    f"neurons only supports 'FirstSpikeTime' readouts")
        
        # Add neuron variable reset
        # **NOTE** we PREPEND the additional variables as
        # they typically depend on the standard variables 
        logger.debug(f"\tReset variables: {additional_reset_vars + genn_model.reset_vars}")
        compile_state.add_neuron_reset_vars(
            pop, additional_reset_vars + genn_model.reset_vars,
            reset_event_ring, reset_v_ring)

        # Add second reset custom update for SHARED_NEURON 
        # variables required by ground truth
        if len(ground_truth.backward_shared_neuron_var_reset) > 0:
            compile_state.add_neuron_reset_vars(
                pop, ground_truth.backward_shared_neuron_var_reset,
                False, False)

        return genn_model
