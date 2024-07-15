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
                         CustomUpdateOnBatchEnd, CustomUpdateOnTimestepEnd)
from ..communicators import Communicator
from ..connection import Connection
from ..losses import Loss, SparseCategoricalCrossentropy, MeanSquareError
from ..neurons import (Input, LeakyIntegrate, LeakyIntegrateFire,
                       LeakyIntegrateFireInput)
from ..optimisers import Optimiser
from ..readouts import AvgVar, AvgVarExpWeight, MaxVar, SumVar, Var
from ..synapses import Exponential
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
from ..utils.network import get_underlying_pop
from ..utils.value import is_value_constant

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

default_params = {
    LeakyIntegrate: {"scale_i": True}, 
    LeakyIntegrateFire: {"relative_reset": False, 
                         "integrate_during_refrac": False,
                         "scale_i": True},
    LeakyIntegrateFireInput: {"relative_reset": False, 
                              "integrate_during_refrac": False,
                              "scale_i": True}}

def _get_tau_syn(pop):
    # Loop through incoming connections
    tau_syn = None
    for conn in pop.incoming_connections:
        # If synapse model isn't exponential, give error
        syn = conn().synapse
        if not isinstance(syn, Exponential):
            raise NotImplementedError(
                "EventProp compiler only supports "
                "Exponential synapses")

        # Update tau_syn
        if tau_syn is None:
            tau_syn = syn.tau
        if tau_syn != syn.tau:
            raise NotImplementedError("EventProp compiler doesn't"
                                      " support neurons whose "
                                      "incoming synapses have "
                                      "different time constants")
    assert tau_syn is not None
    return tau_syn

class CompileState:
    def __init__(self, losses, readouts, backend_name):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self.backend_name = backend_name
        self.weight_optimiser_connections = []
        self._neuron_reset_vars = []
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []
        self.batch_softmax_populations = []
        self.timestep_softmax_populations = []
        self.feedback_connections = []
        self.update_trial_pops = []

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
    def is_reset_custom_update_required(self):
        return len(self._neuron_reset_vars) > 0


class UpdateTrial(Callback):
    def __init__(self, genn_pop):
        self.genn_pop = genn_pop

    def on_batch_begin(self, batch: int):
        # Set dynamic parameter to batch ID
        self.genn_pop.set_dynamic_param_value("Trial", batch)

# Callback which clamps in_syn to 0 one timestep before  
# trial end to avoid bleeding spikes into the next trial
class ZeroInSynLastTimestep(Callback):
    def __init__(self, genn_syn_pop, example_timesteps: int):
        self.genn_syn_pop = genn_syn_pop
        self.example_timesteps = example_timesteps

    def on_timestep_begin(self, timestep: int):
        if timestep == (self.example_timesteps - 1):
            self.genn_syn_pop.out_post.view[:]= 0.0
            self.genn_syn_pop.out_post.push_to_device()

# Standard EventProp weight update model
# **NOTE** feedback is added if required
weight_update_model = {
    "params": [("TauSyn", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("Gradient", "scalar")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "post_neuron_var_refs": [("LambdaI_post", "scalar")],
                             
    "pre_spike_syn_code": """
    addToPost(g);
    """,
    "pre_event_threshold_condition_code": """
    BackSpike_pre
    """,
    "pre_event_syn_code": """
    Gradient -= (LambdaI_post * TauSyn);
    """}

# EventProp weight update model used on KERNEL weights
# **NOTE** feedback is added if required
weight_update_model_kernel = {
    "params": [("TauSyn", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("Gradient", "scalar")],
    "pre_neuron_var_refs": [("BackSpike_pre", "uint8_t")],
    "post_neuron_var_refs": [("LambdaI_post", "scalar")],

    "pre_spike_syn_code": """
    addToPost(g);
    """,
    "pre_event_threshold_condition_code": """
    BackSpike_pre
    """,
    "pre_event_syn_code": """
    atomic_add_Gradient(-(LambdaI_post * TauSyn));
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

gradient_batch_reduce_model = {
    "vars": [("ReducedGradient", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
    "var_refs": [("Gradient", "scalar")],
    "update_code": """
    ReducedGradient = Gradient;
    Gradient = 0;
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
        
    """

    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 reg_lambda_upper: float = 0.0, reg_lambda_lower: float = 0.0,
                 reg_nu_upper: float = 0.0, max_spikes: int = 500,
                 strict_buffer_checking: bool = False,
                 per_timestep_loss: bool = False, dt: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 communicator: Communicator = None, **genn_kwargs):
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

    def pre_compile(self, network: Network, 
                    genn_model, **kwargs) -> CompileState:
        # Build list of output populations
        readouts = [p for p in network.populations
                    if p.neuron.readout is not None]

        return CompileState(self.losses, readouts,
                            genn_model.backend_name)

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state: CompileState) -> NeuronModel:
        # Make copy of model
        model_copy = deepcopy(model)

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
                
            # If neuron model is a leaky integrator
            if isinstance(pop.neuron, LeakyIntegrate):
                # Get tau_syn from population's incoming connections
                tau_syn = _get_tau_syn(pop)

                # Add adjoint state variables
                model_copy.add_var("LambdaV", "scalar", 0.0)
                model_copy.add_var("LambdaI", "scalar", 0.0)

                # Add parameter with synaptic decay constant
                model_copy.add_param("Beta", "scalar", np.exp(-self.dt / tau_syn))

                # Add parameter for scaling factor
                tau_mem = pop.neuron.tau_mem
                model_copy.add_param("TauM", "scalar", tau_mem)
                model_copy.add_param("A", "scalar", 
                                     tau_mem / (tau_mem - tau_syn))

                # Add dynamic parameter to contain trial index and add 
                # population to list of those which require it updating
                model_copy.add_param("Trial", "unsigned int", 0)
                model_copy.set_param_dynamic("Trial")
                compile_state.update_trial_pops.append(pop)

                # Prepend standard code to update LambdaV and LambdaI
                model_copy.prepend_sim_code(
                    f"""
                    LambdaI = drive + ((LambdaI - drive) * Beta) + (A * (LambdaV - drive) * (Alpha - Beta));
                    LambdaV = drive + ((LambdaV - drive) * Alpha);
                    """)

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
                    const float backT = {self.example_timesteps * self.dt} - t - dt;

                    // Backward pass
                    scalar drive = 0.0;
                    """)

                # Add second reset custom update to reset YTrueBack to YTrue
                # **NOTE** seperate as these are SHARED_NEURON variables
                if sce_loss:
                    compile_state.add_neuron_reset_vars(
                        pop, [("YTrueBack", "uint8_t", "YTrue")], False, False)
            # Otherwise, neuron type is unsupported
            else:
                raise NotImplementedError(
                    f"EventProp compiler doesn't support "
                    f"{type(pop.neuron).__name__} neurons")

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
                # Add additional input variable to receive feedback
                model_copy.add_additional_input_var("RevISyn", "scalar", 0.0)

                # If neuron model is LIF
                if isinstance(pop.neuron, LeakyIntegrateFire):
                    # Check EventProp constraints
                    if pop.neuron.integrate_during_refrac:
                        logger.warning("EventProp learning works best with "
                                       "LIF neurons which do not continue to "
                                       "integrate in their refractory period")
                    if pop.neuron.relative_reset:
                        logger.warning("EventProp learning works best "
                                       "with LIF neurons with an "
                                       "absolute reset mechanism")

                    # Get tau_syn from population's incoming connections
                    tau_syn = _get_tau_syn(pop)

                    # Add parameter with synaptic decay constant
                    model_copy.add_param("Beta", "scalar",
                                         np.exp(-self.dt / tau_syn))

                    # Add adjoint state variables
                    model_copy.add_var("LambdaV", "scalar", 0.0)
                    model_copy.add_var("LambdaI", "scalar", 0.0)

                    # Add EGP for IMinusV ring variables
                    model_copy.add_egp("RingIMinusV", "scalar*", 
                                       np.empty(ring_size, dtype=np.float32))

                    # Add parameter for scaling factor
                    tau_mem = pop.neuron.tau_mem
                    model_copy.add_param("A", "scalar", 
                                         tau_mem / (tau_syn - tau_mem))

                    # On backward pass transition, update LambdaV
                    transition_code = """
                        LambdaV += (1.0 / RingIMinusV[ringOffset + RingReadOffset]) * (Vthresh * LambdaV + RevISyn);
                        """

                    # List of variables aside from those in base 
                    # model we want to reset every batch
                    additional_reset_vars = [("LambdaV", "scalar", 0.0),
                                             ("LambdaI", "scalar", 0.0)]

                    # If regularisation is enabled
                    # **THINK** is this LIF-specific?
                    if self.regulariser_enabled:
                        # Add state variables to hold spike count 
                        # during forward and backward pass
                        model_copy.add_var("SpikeCount", "int", 0)
                        model_copy.add_var("SpikeCountBack", "int", 0,
                                           VarAccess.READ_ONLY_DUPLICATE)

                        # Add parameters for regulariser
                        model_copy.add_param("RegNuUpper", "int",
                                             self.reg_nu_upper)
                        model_copy.add_param(
                            "RegLambdaUpper", "scalar",
                            self.reg_lambda_upper / self.full_batch_size)
                        model_copy.add_param(
                            "RegLambdaLower", "scalar",
                            self.reg_lambda_lower / self.full_batch_size)

                        # Add reset variables to copy SpikeCount
                        # into SpikeCountBack and zero SpikeCount
                        additional_reset_vars.extend(
                            [("SpikeCountBack", "int", "SpikeCount"),
                             ("SpikeCount", "int", 0)])

                        # Add additional transition code to apply regularisation
                        transition_code += """
                        if (SpikeCountBack > RegNuUpper) {
                            LambdaV -= RegLambdaUpper * (SpikeCountBack - RegNuUpper);
                        }
                        else {
                            LambdaV -= RegLambdaLower * (SpikeCountBack - RegNuUpper);
                        }
                        """

                        # Add code to update SpikeCount in forward reset code
                        model_copy.append_reset_code("SpikeCount++;")

                    # Add reset logic to reset adjoint state variables 
                    # as well as any state variables from the original model
                    compile_state.add_neuron_reset_vars(
                        pop, model.reset_vars + additional_reset_vars,
                        True, False)

                    # Add code to start of sim code to run backwards pass 
                    # and handle back spikes with correct LIF dynamics
                    model_copy.prepend_sim_code(
                        neuron_backward_pass.substitute(
                            max_spikes=self.max_spikes,
                            example_time=(self.example_timesteps * self.dt),
                            dynamics="""
                            LambdaI = (A * LambdaV * (Beta - Alpha)) + (LambdaI * Beta);
                            LambdaV *= Alpha;
                            """,
                            transition=transition_code))

                    # Prepend (as it accesses the pre-reset value of V) 
                    # code to reset to write spike time and I-V to ring buffer
                    model_copy.prepend_reset_code(
                        neuron_reset.substitute(
                            max_spikes=self.max_spikes,
                            write="RingIMinusV[ringOffset + RingWriteOffset] = Isyn - V;",
                            strict_check=(neuron_reset_strict_check 
                                          if self.strict_buffer_checking
                                          else "")))
                # Otherwise, neuron type is unsupported
                else:
                    raise NotImplementedError(
                        f"EventProp compiler doesn't support "
                        f"{type(pop.neuron).__name__} neurons")

        # Build neuron model and return
        return model_copy

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state: CompileState) -> SynapseModel:
        # **NOTE** this is probably not necessary as 
        # it's also checked in build_neuron_model
        if not isinstance(conn.synapse, Exponential):
            raise NotImplementedError("EventProp compiler only "
                                      "supports Exponential synapses")

        # Return model
        return model

    def build_weight_update_model(self, conn: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state: CompileState) -> WeightUpdateModel:
        if not is_value_constant(connect_snippet.delay):
            raise NotImplementedError("EventProp compiler only "
                                      "support heterogeneous delays")

        # If this is some form of trainable connectivity
        if connect_snippet.trainable:
            # **NOTE** this is probably not necessary as 
            # it's also checked in build_neuron_model
            if isinstance(conn.synapse, Exponential):
                tau_syn = conn.synapse.tau
            else:
                raise NotImplementedError("EventProp compiler only "
                                          "supports Exponential synapses")

            # Determine whether kernel updates are required
            use_kernel = (
                connect_snippet.matrix_type & SynapseMatrixWeight.KERNEL)
            
            # Create basic weight update model
            wum = WeightUpdateModel(
                model=deepcopy(weight_update_model_kernel if use_kernel
                               else weight_update_model),
                param_vals={"TauSyn": tau_syn},
                var_vals={"g": connect_snippet.weight, "Gradient": 0.0},
                pre_neuron_var_refs={"BackSpike_pre": "BackSpike"},
                post_neuron_var_refs={"LambdaI_post": "LambdaI"})
            # Add weights to list of checkpoint vars
            compile_state.checkpoint_connection_vars.append((conn, "g"))

            # Add connection to list of connections to optimise
            compile_state.weight_optimiser_connections.append(conn)
        # Otherwise, e.g. it's a pooling layer
        else:
            wum = WeightUpdateModel(
                model=deepcopy(static_weight_update_model),
                param_vals={"g": connect_snippet.weight},
                pre_neuron_var_refs={"BackSpike_pre": "BackSpike"})

        # If source neuron isn't an input neuron
        source_neuron = conn.source().neuron
        if not isinstance(source_neuron, Input):
            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

            # If it's LIF, add additional event code to backpropagate gradient
            if isinstance(source_neuron, LeakyIntegrateFire):
                wum.add_post_neuron_var_ref("LambdaV_post", "scalar", "LambdaV")
                wum.append_pre_event_syn_code("addToPre(g * (LambdaV_post - LambdaI_post));")

        # Return weight update model
        return wum

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict, 
                                compile_state: CompileState) -> CompiledTrainingNetwork:
        # Correctly target feedback
        for c in compile_state.feedback_connections:
            connection_populations[c].pre_target_var = "RevISyn"

        # Add optimisers to connection weights that require them
        optimiser_custom_updates = []
        for i, c in enumerate(compile_state.weight_optimiser_connections):
            genn_pop = connection_populations[c]
            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Weight{i}", create_wu_var_ref(genn_pop, "g"),
                    create_wu_var_ref(genn_pop, "Gradient"), genn_model))

        # Add per-batch softmax custom updates for each population that requires them
        for i, (p, i, o) in enumerate(compile_state.batch_softmax_populations):
            genn_pop = neuron_populations[p]
            self.add_softmax_custom_updates(genn_model, genn_pop,
                                            i, o, "Batch")

        # Add per-timestep softmax custom updates for each population that requires them
        for i, (p, i) in enumerate(compile_state.timestep_softmax_populations):
            # Create custom update model to implement 
            # first softmax pass and add to model
            genn_pop = neuron_populations[p]
            self._add_softmax_buffer_custom_updates(genn_model, genn_pop, i)

        # Create custom updates to implement variable reset
        compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations)

        # Build list of base callbacks
        base_train_callbacks = []
        base_validate_callbacks = []
        if len(optimiser_custom_updates) > 0:
            if self.full_batch_size > 1:
                base_train_callbacks.append(
                    CustomUpdateOnBatchEnd("GradientBatchReduce"))
            base_train_callbacks.append(
                CustomUpdateOnBatchEnd("GradientLearn"))

        # Add callbacks to set Trial extra global parameter 
        # on populations which require it
        for p in compile_state.update_trial_pops:
            base_train_callbacks.append(UpdateTrial(neuron_populations[p]))

        # Add callbacks to zero insyn on all connections
        # **NOTE** it would be great to be able to do this on device
        for genn_syn_pop in connection_populations.values():
            base_train_callbacks.append(
                ZeroInSynLastTimestep(genn_syn_pop, self.example_timesteps))
            base_validate_callbacks.append(
                ZeroInSynLastTimestep(genn_syn_pop, self.example_timesteps))

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

        # Add reset custom updates
        if compile_state.is_reset_custom_update_required:
            base_train_callbacks.append(CustomUpdateOnBatchBegin("Reset"))
            base_validate_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations,
            self.communicator, compile_state.losses, self._optimiser,
            self.example_timesteps, base_train_callbacks,
            base_validate_callbacks, optimiser_custom_updates,
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
                                        gradient_ref, genn_model):
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
            optimiser_model = self._optimiser.get_model(reduced_gradient, 
                                                        var_ref, False)
        # Otherwise
        else:
            # Create optimiser model with gradient zeroing 
            # logic, connected directly to population
            optimiser_model = self._optimiser.get_model(
                gradient_ref, var_ref, True)

        # Add GeNN custom update to model
        return self.add_custom_update(genn_model, optimiser_model,
                                      "GradientLearn",
                                      "CUGradientLearn" + name_suffix)
