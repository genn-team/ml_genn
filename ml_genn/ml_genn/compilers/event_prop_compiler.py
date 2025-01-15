import logging
import numpy as np

from string import Template
from typing import Iterator, Sequence
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
from ..neurons import (Input, AutoNeuron)
from ..optimisers import Optimiser
from ..readouts import AvgVar, AvgVarExpWeight, MaxVar, SumVar, Var
from ..synapses import AutoSyn
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType
from ..utils.model import (CustomUpdateModel, NeuronModel, 
                           SynapseModel, WeightUpdateModel)
from ..utils.network import PopulationType
from ..utils.snippet import ConnectivitySnippet

from copy import deepcopy
from pygenn import create_egp_ref, create_var_ref, create_wu_var_ref
from .compiler import create_reset_custom_update
from ..utils.module import get_object, get_object_mapping
from ..utils.network import get_underlying_conn, get_underlying_pop
from ..utils.value import is_value_array, is_value_constant

from .compiler import softmax_1_model, softmax_2_model
from ..optimisers import default_optimisers
from ..losses import default_losses
from ..auto_tools import *

logger = logging.getLogger(__name__)

DEBUG = True

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

default_params = {}
"""
default_params = {
    LeakyIntegrate: {"scale_i": True}, 
    LeakyIntegrateFire: {"relative_reset": False, 
                         "integrate_during_refrac": False,
                         "scale_i": True},
    LeakyIntegrateFireInput: {"relative_reset": False, 
                              "integrate_during_refrac": False,
                              "scale_i": True},
    Exponential: {"scale_i": True}}
"""


class CompileState:
    def __init__(self, losses, readouts, backend_name):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self.backend_name = backend_name
        self._optimiser_connections = []
        self._neuron_reset_vars = []
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []
        self.spike_count_populations = []
        self.batch_softmax_populations = []
        self.timestep_softmax_populations = []
        self.feedback_connections = []
        self.update_trial_pops = []

    def add_optimiser_connection(self, conn, weight, delay):
        self._optimiser_connections.append((conn, weight, delay))

    def add_neuron_reset_vars(self, pop, reset_vars, 
                              reset_event_ring, reset_v_ring):
        self._neuron_reset_vars.append((pop, reset_vars, 
                                        reset_event_ring, reset_v_ring))

    def create_reset_custom_updates(self, compiler, genn_model,
                                    neuron_pops):
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
    
    @property
    def optimiser_connections(self):
        return self._optimiser_connections

    @property
    def is_reset_custom_update_required(self):
        return len(self._neuron_reset_vars) > 0


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


# Preprocessing of auto-adjoint equations

def process_odes(vars, params, eqns):
    sym = get_symbols(vars, params)
    dx_dt = {}
    for var in vars:
        if var in eqns:
            dx_dt[var] = parse_expr(eqns[var],local_dict= sym)
    return sym, dx_dt

def process_jumps(sym, jumps):
    h = {}
    for var in jumps:
        tmp = parse_expr(jumps[var],local_dict= sym)-sym[var]
        if sympy.diff(tmp, sym[var]) == 0:
            if tmp != 0:
                h[var] = tmp
        else:
            raise NotImplementedError(
                "EventProp compiler only supports "
                "synapses which (only) add input to target variables.")
    return h

# Standard EventProp weight update model
# **NOTE** feedback is added if required
    
def weight_update_model(syn):
    model= {}
    model["params"] = [ (p[0], p[1]) for p in syn.params ]
    model["vars"] = []
    model["vars"].append(("Gradient","scalar"))
    model["vars"].append((syn.w_name, "scalar", VarAccess.READ_ONLY))
    model["pre_neuron_var_refs"] = [("BackSpike_pre", "uint8_t")]
    model["pre_spike_syn_code"] = f"""
    addToPost({syn.w_name});
    """
    model["pre_event_threshold_condition_code"] = """
    BackSpike_pre
    """
    return model

# TODO: make versions for delayed synapses ...

# Standard EventProp weight update model with fixed delay
# **NOTE** feedback is added if required
delay_weight_update_model = {
    "params": [("TauSyn", "scalar"), ("d", "uint8_t")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("Gradient", "scalar")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "post_neuron_var_refs": [("LambdaI_post", "scalar")],
                             
    "pre_spike_syn_code": """
    addToPostDelay(g, d);
    """,
    "pre_event_threshold_condition_code": """
    BackSpike_pre
    """,
    "pre_event_syn_code": """
    Gradient -= (LambdaI_post[d] * TauSyn);
    """}

# Standard EventProp weight update model with learnable delay
# **NOTE** feedback is added if required
learnable_delay_weight_update_model = {
    "params": [("TauSyn", "scalar"), ("MaxDelay", "int")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("Gradient", "scalar"),
             ("d", "scalar", VarAccess.READ_ONLY), ("DelayGradient", "scalar")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "post_neuron_var_refs": [("LambdaI_post", "scalar"), ("LambdaV_post", "scalar")],
                             
    "pre_spike_syn_code": """
    const int delay = max(0, min(MaxDelay, (int)round(d)));
    addToPostDelay(g, delay);
    """,
    "pre_event_threshold_condition_code": """
    BackSpike_pre
    """,
    "pre_event_syn_code": """
    const int delay = max(0, min(MaxDelay, (int)round(d)));
    Gradient -= (LambdaI_post[delay] * TauSyn);
    DelayGradient -= g * (LambdaI_post[delay] - LambdaV_post[delay]);
    """}

# Weight update model used on non-trainable connections
# **NOTE** feedback is added if required
static_weight_update_model = {
    "params": [("g", "scalar")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "pre_spike_syn_code":
        """
        addToPost(g);
        """,
    "pre_event_threshold_condition_code": """
    BackSpike_pre
    """}

# Weight update model used on non-trainable connections with delay
# **NOTE** feedback is added if required
static_delay_weight_update_model = {
    "params": [("g", "scalar"), ("d", "uint8_t")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "pre_spike_syn_code":
        """
        addToPostDelay(g, d);
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
    if (RingReadOffset != RingReadEndOffset && fabs(backT - RingSpikeTime[ringOffset + RingReadOffset] - dt) < 1e-3*dt) {
        BackSpike = true;
    }

    // Forward pass
    """)


# Template used to generate reset code for neurons
neuron_reset = Template(
    """
    if(RingWriteOffset != RingReadEndOffset) {
        // Write spike time and needed vars to tape
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
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
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
                                                kernel_profiling,
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
        # If maximum delay steps is specified
        if conn.max_delay_steps is not None:
            max_delay_steps = conn.max_delay_steps
        # Otherwise, if delay is constant
        elif is_value_constant(delay):
            max_delay_steps = 1 + delay
        # Otherwise, if delays are specified as an array,
        # calculate maximum delay steps from array
        elif is_value_array(delay):
            max_delay_steps = np.amax(delay) + 1
        else:
            raise RuntimeError(f"Maximum delay associated with Connection "
                              f"{conn.name} cannot be determined "
                              f"automatically, please set max_delay_steps")
    
        # If maximum delay steps is within 8-bit limit, set max delay steps
        # **NOTE** wouldn't be too hard to pick appropriate type
        if max_delay_steps > 255:
            raise NotImplmentedError(f"Maximum of {conn.max_delay_steps} "
                                     f"delay steps for Connection "
                                     f"{conn.name} exceeds 255")
        genn_pop.max_dendritic_delay_timesteps = max_delay_steps

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state: CompileState) -> NeuronModel:
        # Make copy of model
        model_copy = deepcopy(model)

        # neuron dynamical equations
        # Neuron should be AutoNeuron
        if not (isinstance(pop.neuron, AutoNeuron) or isinstance(pop.neuron, Input)):
            raise NotImplementedError(
                f"EventProp compiler only supports "
                f"Input neurons and user-defined neurons (AutoNeuron)")
        if isinstance(pop.neuron, AutoNeuron):
            vars = pop.neuron.vars
            varnames = pop.neuron.varnames
            var_vals = pop.neuron.var_vals
            params = pop.neuron.params
            pnames = pop.neuron.pnames
            param_vals = pop.neuron.param_vals
            n_varnames= varnames.copy()
            vn = varnames.copy()
            vn.append("I")
            sym, dx_dt = process_odes(vn, pnames, pop.neuron.ode)
            saved_vars = set()
            if DEBUG:
                print(f"varnames: {varnames}")
                print(f"var_vals: {var_vals}")
                print(f"pnames: {pnames}")
                print(f"param_vals: {param_vals}")
                print(f"neuron ODEs: {dx_dt}")

            # Add adjoint state variables - only for vars that have an ode
            for var in dx_dt:
                adj_var = f"Lambda{var}"
                sym[adj_var] = sympy.Symbol(adj_var)
                model_copy.add_var(adj_var, "scalar", 0.0)
            # generate adjoint ODE
            # assume that neuron variables do not appear in rhs of post-synapse ODEs
            # therefore, we can do these independent of synapse equations
            dl_dt = {}
            for var in dx_dt:
                o = None
                for v2, expr in dx_dt.items():
                    o = add(o, sympy.diff(expr, sym[var])*sym[f"Lambda{v2}"])    
                dl_dt[f"Lambda{var}"] = o
                # collect variables they might need to go into a ring buffer:
                for v2 in dx_dt:
                    if o.has(sym[v2]):
                        saved_vars.add(v2)

            if DEBUG:
                print(f"Adjoint ODE: {dl_dt}")
            # threshold condition
            if pop.neuron.threshold == "":
                g = None
            else:
                g = parse_expr(pop.neuron.threshold)
            # reset function
            f = {}
            for var in pop.neuron.reset:
                f[var] = parse_expr(pop.neuron.reset[var],local_dict= sym)
            if DEBUG:
                print(f"f: {f}")      
            # after jump dynamics equation "\dot{x}^+"
            dx_dtplusn = {}
            for var, expr in dx_dt.items():
                plus = expr
                for v2, f2 in f.items():
                    plus = plus.subs(sym[v2],f2)
                dx_dtplusn[var] = plus
            if DEBUG:
                print(f"dx_dtplusn: {dx_dtplusn}")
            A = {}
            B = {}
            C = {}
            for var in varnames:
                ex= None
                for v2,f2 in f.items():
                    ex = add(ex, sympy.diff(f2,sym[var])*sym[f"Lambda{v2}"])
                A[var] = ex
                for v2 in varnames:
                    if A[var].has(sym[v2]):
                        saved_vars.add(v2)
                ex= None
                if g is not None:
                    for v2 in varnames:
                        ex = add(ex, sympy.diff(g,sym[v2])*dx_dt[v2])
                    ex = sympy.simplify(ex)
                    if ex != 0:
                        ex = sympy.diff(g,sym[var])/ex
                        if ex != 0:
                            B[var]= simplify_using_threshold(varnames, sym, g, ex)
                            for v2 in varnames:
                                if B[var].has(sym[v2]):
                                    saved_vars.add(v2)
                    if var in B:
                        ex= None
                        for v2,f2 in f.items():
                            ex2= None
                            for v3 in dx_dt:
                                ex2 = add(ex2, sympy.diff(f2,sym[v3])*dx_dt[v3])
                            ex2 = add(ex2,-dx_dtplusn[v2])
                            ex = add(ex,-sym[f"Lambda{v2}"]*ex2)
                        ex = sympy.simplify(ex)
                        C[var] = simplify_using_threshold(varnames, sym, g, ex)
                        for v2 in varnames:
                            if C[var].has(sym[v2]):
                                saved_vars.add(v2)

            if DEBUG:
                print(f"A: {A}")
                print(f"B: {B}")
                print(f"C: {C}")

               
                # Add additional input variable to receive add_to_pre feedback
                model_copy.add_additional_input_var("RevISyn", "scalar", 0.0)
                
                # Add EGP for stored vars ring variables
                ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
                for var in saved_vars:
                    model_copy.add_egp(f"Ring{var}", "scalar*", 
                                   np.empty(ring_size, dtype=np.float32))

                # On backward pass transition, update b_jumps and add_to_pre input

            # assemble the different lambda jump parts
            trans_code= []
            for var in varnames:
                ex = A[var]
                if var in B and var in C:
                    ex2 = C[var]
                else:
                    ex2 = 0
                if var in B:
                    if len(pop.outgoing_connections) > 0:
                        jump = ex + B[var]*(ex2+sympy.Symbol("RevISyn"))
                    else:
                        jump = ex + B[var]*ex2
                else:
                    jump = ex
                jump =  simplify_using_threshold(varnames, sym, g, {"j": jump})["j"]
                for v2 in varnames:
                    if jump.has(sym[v2]):
                        saved_vars.add(v2)
                ccode= sympy.ccode(jump)
                for v2 in saved_vars:
                    ccode= ccode.replace(v2, f"Ring{v2}[ringOffset + RingReadOffset]")
                
                if DEBUG:
                    print(f"jump: {ccode}")

                if jump != 0:
                    trans_code.append(f"Lambda{var} += {ccode};")
            transition_code = "\n".join(trans_code)
            print(f"transition_code: {transition_code}")
            
        # If population has a readout i.e. it's an output
        if pop.neuron.readout is not None:
            sce_loss = isinstance(compile_state.losses[pop], SparseCategoricalCrossentropy)
            mse_loss = isinstance(compile_state.losses[pop], MeanSquareError)
            # Check loss function is compatible
            # **TODO** categorical crossentropy i.e. one-hot encoded
            if not (sce_loss or mse_loss):
                raise NotImplementedError(
                    f"EventProp compiler doesn't support "
                    f"{type(loss).__name__} loss")

            # Add output logic to model
            model_copy = pop.neuron.readout.add_readout_logic(
                model_copy, max_time_required=True, dt=self.dt,
                example_timesteps=self.example_timesteps)

            # **HACK** we don't want to call add_to_neuron on loss function as
            # it will add unwanted code to end of neuron but we do want this
            if sce_loss:
                # Add variable, shared across neurons to hold true label for batch
                model_copy.add_var("YTrue", "uint8_t", 0, 
                                   VarAccess.READ_ONLY_SHARED_NEURON)

                # Add second variable to hold the true label for the backward pass
                model_copy.add_var("YTrueBack", "uint8_t", 0, 
                                   VarAccess.READ_ONLY_SHARED_NEURON)
            elif mse_loss:
                # The true label is the desired voltage output over time
                flat_shape = np.prod(pop.shape)
                egp_size = (self.example_timesteps * self.batch_size * flat_shape)
                model_copy.add_egp("YTrue", "scalar*",
                              np.empty(egp_size, dtype=np.float32))
                
                
            # Add dynamic parameter to contain trial index and add 
            # population to list of those which require it updating
            model_copy.add_param("Trial", "unsigned int", 0)
            model_copy.set_param_dynamic("Trial")
            compile_state.update_trial_pops.append(pop)

            # Prepend standard code to update LambdaV and LambdaI
            # We assume that the neuron model contains a variable "V" that will
            # receive dl_V/dV
            drive = sympy.Symbol("drive")
            dl_dt["LambdaV"] = dl_dt["LambdaV"]+drive

            # If we want to calculate mean-squared error or per-timestep loss
            if self.per_timestep_loss or mse_loss:
                # Get reset vars before we add ring-buffer variables
                reset_vars = model_copy.reset_vars

                # Add variables to hold offsets for 
                # reading and writing ring variables
                model_copy.add_var("RingWriteOffset", "int", 0)
                model_copy.add_var("RingReadOffset", "int", 0)

                # Add EGP for softmax V (SCE) or regression difference (MSE) ring variable
                ring_size = self.batch_size * np.prod(pop.shape) * 2 * self.example_timesteps
                model_copy.add_egp("RingOutputLossTerm", "scalar*", 
                                       np.empty(ring_size, dtype=np.float32))

                if sce_loss:
                    # If readout is AvgVar or SumVar
                    if isinstance(pop.neuron.readout, (AvgVar, SumVar)):
                        model_copy.prepend_sim_code(
                            f"""
                            const int ringOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                            if (Trial > 0) {{
                            RingReadOffset--;
                            const scalar softmax = RingOutputLossTerm[ringOffset + RingReadOffset];
                            const scalar g = (id == YTrueBack) ? (1.0 - softmax) : -softmax;
                            drive = g / (TauM * num_batch * {self.dt * self.example_timesteps});
                            }}
                            
                            // Forward pass
                            """)

                        # Add custom updates to calculate 
                        # softmax from V and write directly to buffermodel_copy
                        compile_state.timestep_softmax_populations.append(
                            (pop, "V"))
                        # Add custom update to reset state
                        compile_state.add_neuron_reset_vars(
                            pop, reset_vars, False, True)
                        
                    # Otherwise, unsupported readout type
                    else:
                        raise NotImplementedError(
                            f"EventProp compiler with CategoricalCrossEntropy loss doesn't support "
                            f"{type(pop.neuron.readout).__name__} readouts")
                elif mse_loss:
                    # Readout has to be Var
                    if isinstance(pop.neuron.readout, Var):
                        model_copy.prepend_sim_code(
                            f"""
                            const int ringOffset = (batch * num_neurons * {2 * self.example_timesteps}) + (id * {2 * self.example_timesteps});
                            if (Trial > 0) {{
                            RingReadOffset--;
                            const scalar error = RingOutputLossTerm[ringOffset + RingReadOffset];
                            drive = error / (TauM * num_batch * {self.dt * self.example_timesteps});
                            }}
                            """)
                        # Add custom update to reset state JAMIE_CHECK
                        compile_state.add_neuron_reset_vars(
                            pop, reset_vars, False, True)
                        # Add code to fill errors into RingBuffer
                        model_copy.append_sim_code(
                            f"""
                            const unsigned int timestep = (int)round(t / dt);
                            const unsigned int index = (batch * {self.example_timesteps} * num_neurons)
                            + (timestep * num_neurons) + id;
                            RingOutputLossTerm[ringOffset + RingWriteOffset] = YTrue[index] - V;
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
                    model_copy.add_var("Softmax", "scalar", 0.0,
                                       VarAccess.READ_ONLY_DUPLICATE)

                    # If readout is AvgVar or SumVar
                    if isinstance(pop.neuron.readout, (AvgVar, SumVar)):
                        model_copy.prepend_sim_code(
                            f"""
                            if (Trial > 0) {{
                            const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                            drive = g / (TauM * num_batch * {self.dt * self.example_timesteps});
                            }}
                            
                            // Forward pass
                            """)
                        
                        # Add custom updates to calculate 
                        # softmax from VSum or VAvg
                        var = ("VSum" if isinstance(pop.neuron.readout, SumVar)
                               else "VAvg")
                        compile_state.batch_softmax_populations.append(
                            (pop, var, "Softmax"))

                        # Add custom update to reset state
                        compile_state.add_neuron_reset_vars(
                            pop, model_copy.reset_vars, False, False)
                    # Otherwise, if readout is AvgVarExpWeight
                    elif isinstance(pop.neuron.readout, AvgVarExpWeight):
                        local_t_scale = 1.0 / (self.dt * self.example_timesteps)
                        model_copy.prepend_sim_code(
                            f"""
                            if (Trial > 0) {{
                            const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                            drive = (g * exp(-(1.0 - (t * {local_t_scale})))) / (TauM * num_batch * {self.dt * self.example_timesteps});
                            }}
                            
                            // Forward pass
                            """)
                        
                        # Add custom updates to calculate softmax from VAvg
                        compile_state.batch_softmax_populations.append(
                            (pop, "VAvg", "Softmax"))

                        # Add custom update to reset state
                        compile_state.add_neuron_reset_vars(
                            pop, model_copy.reset_vars, False, False)
                    # Otherwise, if readout is MaxVar
                    elif isinstance(pop.neuron.readout, MaxVar):
                        # Add state variable to hold vmax from previous trial
                        model_copy.add_var("VMaxTimeBack", "scalar", 0.0,
                                           VarAccess.READ_ONLY_DUPLICATE)

                        model_copy.prepend_sim_code(
                            f"""
                            if (Trial > 0 && fabs(backT - VMaxTimeBack) < 1e-3*dt) {{
                            const scalar g = (id == YTrueBack) ? (1.0 - Softmax) : -Softmax;
                            drive = g / (TauM * num_batch * {self.dt * self.example_timesteps});
                            }}
                            
                            // Forward pass
                            """)
                        
                        # Add custom updates to calculate softmax from VMax
                        compile_state.batch_softmax_populations.append(
                            (pop, "VMax", "Softmax"))

                        # Add custom update to reset state
                        # **NOTE** reset VMaxTimeBack first so VMaxTime isn't zeroed
                        # **TODO** time type
                        compile_state.add_neuron_reset_vars(
                            pop, 
                            [("VMaxTimeBack", "scalar", "VMaxTime")] + model_copy.reset_vars, 
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
                model_copy.prepend_sim_code(
                    f"""
                    // Backward pass
                    scalar drive = 0.0;
                    """)

                # Add second reset custom update to reset YTrueBack to YTrue
                # **NOTE** seperate as these are SHARED_NEURON variables
                if sce_loss:
                    compile_state.add_neuron_reset_vars(
                        pop, [("YTrueBack", "uint8_t", "YTrue")], False, False)

        # Otherwise, it's either an input or a hidden neuron
        # i.e. it requires a ring-buffer
        else:
            # Add variables to hold offsets for 
            # reading and writing ring variables
            model_copy.add_var("RingWriteOffset", "int", 0)
            model_copy.add_var("RingReadOffset", "int", self.max_spikes - 1)

            # Add variables to hold offsets where this neuron
            # started writing to ring during the forward
            # pass and where data to read during backward pass ends
            model_copy.add_var("RingWriteStartOffset", "int",
                               self.max_spikes - 1)
            model_copy.add_var("RingReadEndOffset", "int", 
                               self.max_spikes - 1)

            # Add variable to hold backspike flag
            model_copy.add_var("BackSpike", "uint8_t", False)

            # Add EGP for spike time ring variables
            ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            model_copy.add_egp("RingSpikeTime", "scalar*", 
                               np.empty(ring_size, dtype=np.float32))

            # If neuron is an input
            if isinstance(pop.neuron, Input):
                # Add reset logic to reset any state 
                # variables from the original model
                compile_state.add_neuron_reset_vars(pop, model.reset_vars,
                                                    True, False)

                # Add code to start of sim code to run 
                # backwards pass and handle back spikes
                model_copy.prepend_sim_code(
                    neuron_backward_pass.substitute(
                        max_spikes=self.max_spikes,
                        example_time=(self.example_timesteps * self.dt),
                        dynamics="",
                        transition=""))

                # Prepend code to reset to write spike time to ring buffer
                model_copy.prepend_reset_code(
                    neuron_reset.substitute(
                        max_spikes=self.max_spikes,
                        write="",
                        strict_check=(neuron_reset_strict_check
                                      if self.strict_buffer_checking
                                      else "")))
            # Otherwise i.e. it's hidden
            else:
                # If regularisation is enabled
                # **THINK** is this LIF-specific?
                if self.regulariser_enabled:
                    # Add state variables to hold spike count
                    # during forward and backward pass. 
                    # **NOTE** SpikeCountBackSum is shared across
                    # batches as it is the result of a reduction
                    model_copy.add_var("SpikeCount", "int", 0)
                    model_copy.add_var("SpikeCountBackBatch", "int", 0,
                                       VarAccess.READ_ONLY)

                    # Add parameters for regulariser
                    # **NOTE** this is multiplied by batch_size so it
                    # can be compared directly to SpikeCountBackBatch
                    model_copy.add_param("RegNuUpperBatch", "int",
                                         self.reg_nu_upper * self.full_batch_size)
                        
                    # **NOTE** these are divided by batch size once to
                    # make these batch-size-agnostic and again to take 
                    # into account that we're operating on batch sums of spike counts
                    model_copy.add_param(
                        "RegLambdaUpper", "scalar",
                        self.reg_lambda_upper / (self.full_batch_size
                                                 * self.full_batch_size))
                    model_copy.add_param(
                        "RegLambdaLower", "scalar",
                        self.reg_lambda_lower / (self.full_batch_size
                                                 * self.full_batch_size))

                    # If batch size is 1, add reset variables to copy SpikeCount
                    # into SpikeCountBackBatch and zero SpikeCount
                    # **NOTE** if batch size > 1, SpikeCountBackBatch is
                    # calculated with a reduction which zeroes SpikeCount
                    if self.full_batch_size == 1:
                        additional_reset_vars.extend(
                            [("SpikeCountBackBatch", "int", "SpikeCount"),
                             ("SpikeCount", "int", 0)])

                    # Add additional transition code to apply regularisation
                    transition_code += """
                    if (SpikeCountBackBatch > RegNuUpperBatch) {
                    LambdaV -= RegLambdaUpper * (SpikeCountBackBatch - RegNuUpperBatch);
                    }
                    else {
                    LambdaV -= RegLambdaLower * (SpikeCountBackBatch - RegNuUpperBatch);
                    }
                    """
                        
                    # Add population to list of those that 
                    # require a spike count reduction
                    compile_state.spike_count_populations.append(pop)

                    # Add code to update SpikeCount in forward reset code
                    model_copy.append_reset_code("SpikeCount++;")

        if not isinstance(pop.neuron, Input) and pop.neuron.readout is None:
            # List of variables aside from those in base 
            # model we want to reset every batch
            additional_reset_vars = [(f"Lambda{var}", "scalar", 0.0) for var in varnames]
            # Add reset logic to reset adjoint state variables 
            # as well as any state variables from the original model
            print(type(pop.neuron))
            compile_state.add_neuron_reset_vars(
                pop, model.reset_vars + additional_reset_vars,
                True, False)
            print(f"model.reset_vars: {model.reset_vars}")
            # Prepend (as it accesses the pre-reset value of V) 
            # code to reset to write spike time and I-V to ring buffer
            # TODO: we need the saved vars here to go into the buffer
            write = []
            for var in saved_vars:
                write.append(f"Ring{var}[ringOffset + RingWriteOffset] = {var};")
            model_copy.prepend_reset_code(
                neuron_reset.substitute(
                    max_spikes=self.max_spikes,
                    write= "\n".join(write),
                    strict_check=(neuron_reset_strict_check 
                                  if self.strict_buffer_checking
                                  else "")))

        if not isinstance(pop.neuron, Input):
            # Add code to start of sim code to run backwards pass 
            # and handle back spikes with correct dynamics
            dt = sympy.Symbol("dt")
            _, clines = solve_ode(pop.neuron.varnames, sym, dl_dt, dt, pop.neuron.solver)
            ccode = "\n".join(clines)
            if pop.neuron.readout is None:
                model_copy.append_sim_code(
                    neuron_backward_pass.substitute(
                        max_spikes=self.max_spikes,
                        example_time=(self.example_timesteps * self.dt),
                        dynamics=ccode,
                        transition=transition_code))
            else:
                model_copy.append_sim_code(f"""
                // Backward pass
                {ccode}
                
                // Forward pass
                """)
            # Add the neuron simcode including all the inherited I ODE equations
            _, clines = solve_ode(varnames, sym, dx_dt, dt, pop.neuron.solver)
            ccode = "\n".join(clines)
            model_copy.append_sim_code(ccode)

        for key,val in model_copy.model.items():
            print(f"\n {key}")
            print(val)
        return model_copy

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state: CompileState) -> SynapseModel:
        # **NOTE** this is probably not necessary as 
        # it's also checked in build_neuron_model
        # **NOTE** this must be run after build_neuron_model!!!
        if not isinstance(conn.synapse, AutoSyn):
            raise NotImplementedError("EventProp compiler only "
                                      "supports user-defined synapses (AutoSyn)")

        # Make copy of model
        model_copy = deepcopy(model)
        syn= conn.synapse
        if not isinstance(syn, AutoSyn):
            raise NotImplementedError(
                "EventProp compiler only supports "
                "user-defined synapses (AutoSyn)")

        # assemble forward and backward pass equations for synaptic ODE
        sym, dx_dt = process_odes(syn.varnames, syn.pnames, syn.ode)

        # synaptic jumps 
        h = process_jumps(sym, syn.jumps)
        if DEBUG:
            print(f"h: {h}")
                
        # generate forward jumps
        clines = []
        for var, expr in h.items():
            clines.append(f"{var} += {sympy.ccode(expr.subs(sympy.Symbol(syn.w_name),sympy.Symbol('inSyn')))};")
        clines.append("inSyn = 0;")
        jump_ccode = "\n".join(clines)
        # generate adjoint ODE
        pop = conn.target()
        vn = pop.neuron.varnames.copy()
        vn.append("I")
        n_sym, n_dx_dt = process_odes(vn, pop.neuron.pnames, pop.neuron.ode)
        dl_dt = {}
        post_var_refs = set()
        params = set()
        for var in dx_dt:
            # add adjoint variable to post-synapse model
            model_copy.model["vars"].append((f"Lambda{var}","scalar"))
            model_copy.var_vals[f"Lambda{var}"] = 0.0
            o = None
            for v2, expr in dx_dt.items():
                o = add(o, sympy.diff(expr, sym[var])*sym[f"Lambda{v2}"])    
            for v2, expr in n_dx_dt.items():
                o = add(o, sympy.diff(expr, sym[var])*n_sym[f"Lambda{v2}"])  
            dl_dt[f"Lambda{var}"] = o
            err = False
            for v2 in syn.varnames:
                if o.has(sym[v2]):
                    err = True
            for v2 in pop.neuron.varnames:
                if o.has(n_sym[v2]):
                    err = True
            if err:
                raise NotImplementedError(
                    f"Equations necessitate saving forward pass variables in a currently not supported setting.")
            for v2 in pop.neuron.varnames:
                print(v2)
                print(o)
                if (o.has(n_sym[v2])):
                    post_var_refs.add((v2, "scalar"))
            for p in pop.neuron.params:
                if (o.has(n_sym[p[0]])):
                    params.add(p)
            print(syn)
            print(syn.param_vals)
        for ref in post_var_refs:
            print(ref)
            model_copy.model["neuron_var_refs"].append(ref) 
        for p in params:
            model_copy.model["params"].append((p[0],p[1]))
            model_copy.param_vals[p[0]] = p[2] 
                    
        dt = sympy.Symbol("dt")
        _, clines = solve_ode(syn.varnames, sym, dx_dt, dt, syn.solver)
        fwd_ccode = "\n".join(clines)
        lbd_names = [ f"Lambda{var}" for var in syn.varnames ]
        _, clines = solve_ode(lbd_names, sym, dl_dt, dt, syn.solver)
        bwd_ccode = "\n".join(clines)
        model_copy.model["sim_code"] = f"""
            // Backward pass
            {bwd_ccode}
            injectCurrent({syn.inject_current});
            // Forward pass
            {jump_ccode}
            {fwd_ccode}
        """
        # Return model
        print("Synapse model")
        for key,val in model_copy.model.items():
            print(f"\n {key}")
            print(val)
        print(dir(model_copy))
        return model_copy

    def build_weight_update_model(self, conn: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state: CompileState) -> WeightUpdateModel:
        # Does this connection have a delay?
        has_delay = (not is_value_constant(connect_snippet.delay)
                     or connect_snippet.delay > 0)
        
        # Does this connection have learnable delays
        has_learnable_delay = conn in self.delay_learn_conns
    
        # If this is some form of trainable connectivity
        if connect_snippet.trainable:
            # **NOTE** this is probably not necessary as 
            # it's also checked in build_neuron_model
            if not isinstance(conn.synapse, AutoSyn):
                raise NotImplementedError("EventProp compiler only "
                                          "supports Auto synapses")

            # Determine whether atomic weight updates are required
            # **YUCK** backend-specific code shouldn't be required in models
            # **TODO** something when OpenCL
            use_atomic = (connect_snippet.matrix_type & SynapseMatrixWeight.KERNEL)
            assert not use_atomic

            """
            # If this connection has learnable delays
            if has_learnable_delay:
                # Check maximum delay steps is set
                if conn.max_delay_steps is None:
                    raise RuntimeError(f"Maximum delay steps must be specified for "
                                       f"Connection {conn.name} with delay learning")

                # Create weight update model with learable delays
                wum = WeightUpdateModel(
                    model=deepcopy(learnable_delay_weight_update_model),
                    param_vals={"TauSyn": tau_syn, "MaxDelay": conn.max_delay_steps},
                    var_vals={"g": connect_snippet.weight, "Gradient": 0.0,
                              "d": connect_snippet.delay, "DelayGradient": 0.0},
                    pre_neuron_var_refs={"BackSpike_pre": "BackSpike"},
                    post_neuron_var_refs={"LambdaI_post": "LambdaI", "LambdaV_post": "LambdaV"})
                
                # Add delays to list of checkpoint vars
                compile_state.checkpoint_connection_vars.append((conn, "d"))

                # Mark connection as requiring delay and weight optimisation
                compile_state.add_optimiser_connection(conn, True, True)
            # Otherwise
            else:
                # Create basic weight update model
                wum = WeightUpdateModel(
                    model=deepcopy(delay_weight_update_model if has_delay 
                                else weight_update_model),
                    param_vals={"TauSyn": tau_syn},
                    var_vals={"g": connect_snippet.weight, "Gradient": 0.0},
                    pre_neuron_var_refs={"BackSpike_pre": "BackSpike"},
                    post_neuron_var_refs={"LambdaI_post": "LambdaI"})

                # Mark connection as requiring weight optimisation
                compile_state.add_optimiser_connection(conn, True, False)

            # Add weights to list of checkpoint vars
            compile_state.checkpoint_connection_vars.append((conn, "g"))

        # Otherwise, e.g. it's a pooling layer
        else:
            wum = WeightUpdateModel(
                model=deepcopy(static_delay_weight_update_model if has_delay 
                               else static_weight_update_model),
                param_vals={"g": connect_snippet.weight},
                pre_neuron_var_refs={"BackSpike_pre": "BackSpike"})
        
        # If connection has non-learnable delay, set parameter value
        # **NOTE** delay is a state variable for learnable connections
        if not has_learnable_delay and has_delay:
            wum.param_vals["d"] = connect_snippet.delay

        # If source neuron isn't an input neuron
        source_neuron = conn.source().neuron
        if not isinstance(source_neuron, Input):
            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

            # If it's LIF, add additional event code to backpropagate gradient
            if isinstance(source_neuron, LeakyIntegrateFire):
                if has_learnable_delay:
                    wum.append_pre_event_syn_code("addToPre(g * (LambdaV_post[delay] - LambdaI_post[delay]));")
                else:
                    wum.add_post_neuron_var_ref("LambdaV_post", "scalar", "LambdaV")
                    
                    if has_delay:
                        wum.append_pre_event_syn_code("addToPre(g * (LambdaV_post[d] - LambdaI_post[d]));")
                    else:
                        wum.append_pre_event_syn_code("addToPre(g * (LambdaV_post - LambdaI_post));")

        # Return weight update model
        """

        source_neuron = conn.source().neuron
        if not isinstance(source_neuron, Input):
            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

        pop = conn.target()
        syn = conn.synapse
        syn.var_vals["Gradient"] = 0.0
        weight_update= weight_update_model(syn)

        post_var_refs= {}
        param_vals = syn.param_vals

        pop = conn.target()
        # forward ODE
        sym, dx_dt = process_odes(syn.varnames, syn.pnames, syn.ode)
        # synaptic jumps 
        h = process_jumps(sym, syn.jumps)        
        if DEBUG:
            print(f"h: {h}")
                
        # assemble gradient update
        grad_update = None
        for var in h:
            if h[var] != 0:
                grad_update = add(grad_update,-sym[f"Lambda{var}"]*sympy.diff(h[var],sympy.Symbol(syn.w_name)))
                post_var_refs[f"Lambda{var}"]= f"Lambda{var}"
                

        weight_update["pre_event_syn_code"] = f"Gradient += {sympy.ccode(grad_update)};"
        print(f"POSTVAR: {post_var_refs}")

        # assemble dx_dt_plusm; 
        # ***NOTE: we here have to work with the POST-SYNAPTIC neurons and their equations
        # this currently only works if the inject_current jumps only depend on w and
        # inject_current only enters linearly into rhs of any ODEs
        # Do dx_dt_plusm
        # synaptic ODEs first
        dx_dtplusm = {}
        for var, expr in dx_dt.items():
            for v2, ex2 in h.items():
                expr = expr.subs(sym[v2],sym[v2]+ex2)
            dx_dtplusm[var] = expr
       
        vn = pop.neuron.varnames.copy()
        vn.append("I")
        n_sym, n_dx_dt = process_odes(vn, pop.neuron.pnames, pop.neuron.ode)
        inject = sympy.parse_expr(syn.inject_current, local_dict=sym)
        inject_plusm = inject
        for var, expr in h.items():
                inject_plusm = inject_plusm.subs(sym[var],sym[var]+expr)
        n_dx_dtplusm = {}
        print(f"n_dx_dt: {n_dx_dt}")
        for var, expr in n_dx_dt.items():
            ex2 = expr.subs(n_sym["I"],inject)
            n_dx_dt[var] = ex2
            ex2 = expr.subs(n_sym["I"],inject_plusm)
            n_dx_dtplusm[var] = ex2
        if DEBUG:
            print(f"dx_dt: {dx_dt}")
            print(f"n_dx_dt: {n_dx_dt}")
            print(f"dx_dtplusm: {dx_dtplusm}")
            print(f"n_dx_dtplusm: {n_dx_dtplusm}")
        # SIMPLIFICATON: no dependencies of synaptic jumps on pre- or post-synaptic
        # variables!
        # add_to_pre is based on the difference of dx_dt and dx_dtplusm 
        ex = None
        # synapse equations first:
        for var, expr in dx_dt.items():
            ex2 = dx_dtplusm[var] - expr
            ex = add(ex, sym[f"Lambda{var}"]*ex2)
        # then neuron equations:
        for var, expr in n_dx_dt.items():
            ex2 = n_dx_dtplusm[var] - expr
            ex = add(ex, sympy.Symbol(f"Lambda{var}")*ex2)
        if ex is not None:
            add_to_pre = sympy.simplify(ex)
            # check whether any vars might be involved from teh forward pass
            err = False
            for var in syn.varnames:
                if add_to_pre.has(sym[var]):
                    err = True
            for var in pop.neuron.varnames:
                if add_to_pre.has(n_sym[var]):
                    err = True
            if err:
                raise NotImplementedError(
                    f"Equations necessitate saving forward pass variables in a currently not supported setting.")
            # put var_refs to post-synaptic vars as needed
            add_to_pre_ccode = sympy.ccode(add_to_pre)
            for p,value in pop.neuron.param_vals.items():
                sym_p = sympy.Symbol(p)
                if add_to_pre.has(sym_p):
                    param_vals[p]= value
                    weight_update["params"].append((p,"scalar"))
            weight_update["post_neuron_var_refs"] = []
            for var,tpe,_ in pop.neuron.vars+syn.vars:
                sym_v = sympy.Symbol(var)
                if add_to_pre.has(sym_v):
                    weight_update["post_neuron_var_refs"].append((var,tpe))
                    post_var_refs[var] = var
                sym_l = sympy.Symbol(f"Lambda{var}")
                if add_to_pre.has(sym_l):
                    weight_update["post_neuron_var_refs"].append((f"Lambda{var}",tpe))
                    post_var_refs[f"Lambda{var}"] = f"Lambda{var}"
                
        wum = WeightUpdateModel(
            model= weight_update,
            param_vals= param_vals,
            var_vals= {syn.w_name: connect_snippet.weight, "Gradient": 0.0},
            pre_neuron_var_refs={"BackSpike_pre": "BackSpike"},
            post_neuron_var_refs=post_var_refs
        )
        wum.append_pre_event_syn_code(f"addToPre({add_to_pre_ccode});")
        for key,val in wum.model.items():
            print(f"\n {key}")
            print(val)
        print(wum.post_neuron_var_refs)
        return wum

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
                    f"Weight{i}", create_wu_var_ref(genn_pop, "g"),
                    create_wu_var_ref(genn_pop, "Gradient"), 
                    self._optimiser, genn_model)
                
                # Add custom update to list of optimisers
                weight_optimiser_cus.append(cu_weight)

                # Add gradient to list of gradient vars to zero
                gradient_vars.append(("Gradient", "scalar", 0.0))
            
            # If delay optimiser is required
            if d:
                # Create delay optimiser custom update
                cu_delay = self._create_optimiser_custom_update(
                    f"Delay{i}", create_wu_var_ref(genn_pop, "d"),
                    create_wu_var_ref(genn_pop, "DelayGradient"),
                    self._delay_optimiser, genn_model,
                    (0.0, c.max_delay_steps))

                # Add custom update to list of optimisers
                delay_optimiser_cus.append(cu_delay)
                
                # Add gradient to list of gradient vars to zero
                gradient_vars.append(("DelayGradient", "scalar", 0.0))

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
                                                  neuron_populations)

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
