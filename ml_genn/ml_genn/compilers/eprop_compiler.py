import logging
import numpy as np

from typing import Iterator, Sequence
from pygenn import CustomUpdateVarAccess, SynapseMatrixType, VarAccess
from . import Compiler
from .compiled_training_network import CompiledTrainingNetwork
from .deep_r import RewiringRecord
from .. import Connection, Population, Network
from ..callbacks import (BatchProgressBar, CustomUpdateOnBatchBegin,
                         CustomUpdateOnBatchEnd, CustomUpdateOnTimestepEnd,
                         CustomUpdateOnTrainBegin)
from ..communicators import Communicator
from ..losses import Loss, SparseCategoricalCrossentropy
from ..metrics import MetricsType
from ..neurons import (AdaptiveLeakyIntegrateFire, Input,
                       LeakyIntegrate, LeakyIntegrateFire, 
                       LeakyIntegrateFireInput)
from ..optimisers import Optimiser
from ..synapses import Delta
from ..utils.callback_list import CallbackList
from ..utils.model import (CustomUpdateModel, NeuronModel,
                           SynapseModel, WeightUpdateModel)
from ..utils.snippet import ConnectivitySnippet

from copy import deepcopy
from pygenn import create_var_ref, create_wu_var_ref
from .compiler import create_reset_custom_update
from .deep_r import add_deep_r
from ..utils.module import get_object, get_object_mapping
from ..utils.network import get_underlying_conn
from ..utils.value import is_value_constant

from ml_genn.optimisers import default_optimisers
from ml_genn.losses import default_losses

logger = logging.getLogger(__name__)

default_params = {
    AdaptiveLeakyIntegrateFire: {"relative_reset": True,
                                 "integrate_during_refrac": True},
    LeakyIntegrate: {"scale_i": False}, 
    LeakyIntegrateFire: {"relative_reset": True, 
                         "integrate_during_refrac": True,
                         "scale_i": False},
    LeakyIntegrateFireInput: {"relative_reset": True, 
                              "integrate_during_refrac": True,
                              "scale_i": False}}

def _has_connection_to_output(pop):
    # Loop through population's outgoing connections
    for c in pop.outgoing_connections:
        # If target of connection has a readout 
        # i.e. it's an output, return true
        if c().target().neuron.readout is not None:
            return True

    return False

class CompileState:
    def __init__(self, losses, readouts):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self._tau_mem = None
        self._tau_adapt = None
        self.feedback_connections = []
        self.weight_optimiser_connections = []
        self.bias_optimiser_populations = []
        self.softmax_populations = []
        self._neuron_reset_vars = {}
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []

    def add_neuron_readout_reset_vars(self, pop):
        reset_vars = pop.neuron.readout.reset_vars
        if len(reset_vars) > 0:
            assert pop not in self._neuron_reset_vars
            self._neuron_reset_vars[pop] = reset_vars

    def create_reset_custom_updates(self, compiler, genn_model,
                                    neuron_pops):
        # Loop through neuron variables to reset
        for i, (pop, reset_vars) in enumerate(self._neuron_reset_vars.items()):
            # Create reset model
            model = create_reset_custom_update(
                reset_vars,
                lambda name: create_var_ref(neuron_pops[pop], name))

            # Add custom update
            compiler.add_custom_update(genn_model, model, 
                                       "Reset", f"CUResetNeuron{i}")

    @property
    def is_reset_custom_update_required(self):
        return len(self._neuron_reset_vars) > 0

    @property
    def tau_mem(self):
        assert self._tau_mem is not None

        return self._tau_mem

    @tau_mem.setter
    def tau_mem(self, tau_mem):
        if self._tau_mem is None:
            self._tau_mem = tau_mem
        if self._tau_mem != tau_mem:
            raise NotImplementedError("E-prop compiler doesn't "
                                      "support neurons with "
                                      "different time constants")

    @property
    def tau_adapt(self):
        assert self._tau_adapt is not None

        return self._tau_adapt

    @tau_adapt.setter
    def tau_adapt(self, tau_adapt):
        if self._tau_adapt is None:
            self._tau_adapt = tau_adapt
        if self._tau_adapt != tau_adapt:
            raise NotImplementedError("E-prop compiler doesn't "
                                      "support neurons with "
                                      "different time constants")

eprop_lif_model = {
    "params": [("CReg", "scalar"), ("Alpha", "scalar"), 
               ("FTarget", "scalar"), ("AlphaFAv", "scalar"),
               ("Vthresh_post", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY),
             ("eFiltered", "scalar"), ("DeltaG", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_vars": [("Psi", "scalar"), ("FAvg", "scalar")],
    "post_neuron_var_refs": [("RefracTime_post", "scalar"), ("V_post", "scalar"),
                             ("E_post", "scalar")],

    "pre_spike_code": """
    ZFilter += 1.0;
    """,
    "pre_dynamics_code": """
    ZFilter *= Alpha;
    """,

    "post_spike_code": """
    FAvg += (1.0 - AlphaFAv);
    """,
    "post_dynamics_code": """
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - Vthresh_post) / Vthresh_post));
    }
    """,

    "pre_spike_syn_code": """
    addToPost(g);
    """,
    "synapse_dynamics_code": """
    const scalar e = ZFilter * Psi;
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;
    DeltaG += (eF * E_post) + ((FAvg - FTarget) * CReg * e);
    eFiltered = eF;
    """}

eprop_alif_model = {
    "params": [("CReg", "scalar"), ("Alpha", "scalar"), ("Rho", "scalar"),
               ("FTarget", "scalar"),("AlphaFAv", "scalar"),
               ("Vthresh_post", "scalar"), ("Beta_post", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY),
             ("eFiltered", "scalar"), ("epsilonA", "scalar"),
             ("DeltaG", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_vars": [("Psi", "scalar"), ("FAvg", "scalar")],
    "post_neuron_var_refs": [("RefracTime_post", "scalar"), ("V_post", "scalar"),
                             ("A_post", "scalar"), ("E_post", "scalar")],
 
    "pre_spike_code": """
    ZFilter += 1.0;
    """,
    "pre_dynamics_code": """
    ZFilter *= Alpha;
    """,

    "post_spike_code": """
    FAvg += (1.0 - AlphaFAv);
    """,
    "post_dynamics_code": """
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh_post + (Beta_post * A_post))) / Vthresh_post));
    }
    """,

    "pre_spike_syn_code": """
    addToPost(g);
    """,
    "synapse_dynamics_code": """
    // Calculate some common factors in e and epsilon update
    scalar epsA = epsilonA;
    const scalar psiZFilter = Psi * ZFilter;
    const scalar psiBetaEpsilonA = Psi * Beta_post * epsA;

    // Calculate e and episilonA
    const scalar e = psiZFilter  - psiBetaEpsilonA;
    epsilonA = psiZFilter + ((Rho * epsA) - psiBetaEpsilonA);

    // Calculate filtered version of eligibility trace
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;
    
    // Apply weight update
    DeltaG += (eF * E_post) + ((FAvg - FTarget) * CReg * e);
    eFiltered = eF;
    """}

output_learning_model = {
    "params": [("Alpha", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("DeltaG", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_neuron_var_refs": [("E_post", "scalar")],

    "pre_spike_code": """
    ZFilter += 1.0;
    """,
    "pre_dynamics_code": """
    ZFilter *= Alpha;
    """,

    "pre_spike_syn_code": """
    addToPost(g);
    """,
    "synapse_dynamics_code": """
    DeltaG += ZFilter * E_post;
    addToPre(g * E_post);
    """}

gradient_batch_reduce_model = {
    "vars": [("ReducedGradient", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
    "var_refs": [("Gradient", "scalar")],
    "update_code": """
    ReducedGradient = Gradient;
    Gradient = 0;
    """}

class EPropCompiler(Compiler):
    """Compiler for training models using e-prop [Bellec2020]_.
    
    The e-prop compiler supports :class:`ml_genn.neurons.LeakyIntegrateFire` and
    :class:`ml_genn.neurons.AdaptiveLeakyIntegrateFire` hidden neuron models; and 
    :class:`ml_genn.losses.SparseCategoricalCrossentropy` loss functions for classification
    and :class:`ml_genn.losses.MeanSquareError` for regression.
    
    e-prop is derived from Real-Time Recurrent Learning (RTRL) so does not require a
    backward pass meaning that its memory overhead does not scale with sequence length.
    However, e-prop requires a per-connection eligibility trace meaning that it is
    incompatible with connectivity like convolutions with shared weights. Furthermore,
    because each connection has to be updated every timestep, training performance is not
    improved by sparse activations.
    
    Args:
        example_timesteps:          How many timesteps each example will be
                                    presented to the network for
        losses:                     Either a dictionary mapping loss functions
                                    to output populations or a single loss
                                    function to apply to all outputs
        optimiser:                  Optimiser to use when applying weights
        tau_reg:                    Time constant with which hidden neuron
                                    spike trains are filtered to obtain the
                                    firing rate used for regularisation [ms]
        c_reg:                      Regularisation strength
        f_target:                   Target hidden neuron firing rate used for
                                    regularisation [Hz]
        train_output_bias:          Should output neuron biases be trained?
        dt:                         Simulation timestep [ms]
        batch_size:                 What batch size should be used for
                                    training? In our experience, e-prop works
                                    well with very large batch sizes (512)
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
                 tau_reg: float = 500.0, c_reg: float = 0.001, 
                 f_target: float = 10.0, train_output_bias: bool = True,
                 dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False,
                 reset_time_between_batches: bool = True,
                 communicator: Communicator = None, 
                 deep_r_exc_conns: Sequence = [],
                 deep_r_inh_conns: Sequence = [],
                 deep_r_l1_strength: float = 0.01,
                 deep_r_record_rewirings = {},
                 **genn_kwargs):
        supported_matrix_types = [SynapseMatrixType.SPARSE,
                                  SynapseMatrixType.DENSE]
        super(EPropCompiler, self).__init__(supported_matrix_types, dt,
                                            batch_size, rng_seed,
                                            kernel_profiling,
                                            communicator,
                                            **genn_kwargs)
        self.example_timesteps = example_timesteps
        self.losses = losses
        self._optimiser = get_object(optimiser, Optimiser, "Optimiser",
                                     default_optimisers)
        self.tau_reg = tau_reg
        self.c_reg = c_reg
        self.f_target = f_target
        self.train_output_bias = train_output_bias
        self.reset_time_between_batches = reset_time_between_batches
        self.deep_r_exc_conns = set(get_underlying_conn(c)
                                    for c in deep_r_exc_conns)
        self.deep_r_inh_conns = set(get_underlying_conn(c)
                                    for c in deep_r_inh_conns)
        self.deep_r_l1_strength = deep_r_l1_strength
        self.deep_r_record_rewirings = deep_r_record_rewirings

    def pre_compile(self, network: Network, 
                    genn_model, **kwargs) -> CompileState:
        # Build list of output populations
        readouts = [p for p in network.populations
                    if p.neuron.readout is not None]

        return CompileState(self.losses, readouts)

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state: CompileState) -> NeuronModel:
        # Make copy of model
        model_copy = deepcopy(model)

        # If population has a readout i.e. it's an output
        if pop.neuron.readout is not None:
            # Get loss function associated with this output neuron
            loss = compile_state.losses[pop]

            # If loss function is sparse categorical crossentropy
            if isinstance(loss, SparseCategoricalCrossentropy):
                # Get output variable from neuron model
                output_var = model_copy.output_var

                # Add softmax variable with same type as 
                # output variable and initialise to zero
                softmax_var_name = output_var[0] + "Softmax"
                model_copy.add_var(softmax_var_name, output_var[1], 0)

                # Finally, point output variable at new softmax'd output
                model_copy.output_var_name = softmax_var_name

                # Add population to list of those needing softmax calculation
                compile_state.softmax_populations.append(
                    (pop, output_var[0], softmax_var_name))

            # Add output logic to model
            model_copy = pop.neuron.readout.add_readout_logic(
                model_copy, example_timesteps=self.example_timesteps,
                dt=self.dt)

            # Add any output reset variables to compile state
            compile_state.add_neuron_readout_reset_vars(pop)

            # Add state variable to hold error
            # **NOTE** all loss functions require this!
            model_copy.add_var("E", "scalar", 0.0)

            # Add loss function to neuron model
            # **THINK** semantics of this i.e. modifying inplace 
            # seem a bit different than others
            loss.add_to_neuron(model_copy, pop.shape, 
                               self.batch_size, self.example_timesteps)

            # Add sim-code to calculate error
            model_copy.append_sim_code(
                f"""
                E = {model_copy.output_var_name} - yTrue;
                """)

            # If we should train output biases
            if self.train_output_bias:
                # Convert Bias parameter into variable
                # **THINK** do we want some sort of duck-type to get bias var
                # or add something to model to say which variable is bias
                model_copy.make_param_var("Bias")

                # Add DeltaBias
                model_copy.add_var("DeltaBias", "scalar", 0.0)

                # Add sim-code to update DeltaBias
                model_copy.append_sim_code("DeltaBias += E;")

                # Add population to list of those with biases to optimise
                compile_state.bias_optimiser_populations.append(pop)

                # Add bias to list of checkpoint vars
                compile_state.checkpoint_population_vars.append((pop, "Bias"))

        # Otherwise, if neuron isn't an input i.e. it's hidden
        elif not isinstance(pop.neuron, Input):
            # Check hidden population is connected directly to output
            if not _has_connection_to_output(pop):
                raise RuntimeError("In models trained with e-prop, all "
                                   "hidden populations must be directly "
                                   "connected to an output population")

            # Add additional input variable to receive feedback
            model_copy.add_additional_input_var("ISynFeedback", "scalar", 0.0)

            # Add state variable to store 
            # feedback and initialise to zero
            model_copy.add_var("E", "scalar", 0.0)

            # Add sim code to store incoming feedback in new state variable
            model_copy.append_sim_code("E = ISynFeedback;")

            # If neuron model is LIF or ALIF
            if isinstance(pop.neuron, (AdaptiveLeakyIntegrateFire,
                                       LeakyIntegrateFire)):
                # Check e-prop constraints
                if not pop.neuron.integrate_during_refrac:
                    logger.warning("E-prop learning works best with (A)LIF "
                                   "neurons which continue to integrate "
                                   "during their refractory period")
                if not pop.neuron.relative_reset:
                    logger.warning("E-prop learning works best with (A)LIF "
                                   "neurons with a relative reset mechanism")

                # Set global time constants from neuron model
                compile_state.tau_mem = pop.neuron.tau_mem
                if isinstance(pop.neuron, AdaptiveLeakyIntegrateFire):
                    compile_state.tau_adapt = pop.neuron.tau_adapt
            else:
                raise NotImplementedError(f"E-prop compiler doesn't support "
                                          f"{type(pop.neuron).__name__} "
                                          f"neurons")

        # Build neuron model and return
        return model_copy

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state: CompileState) -> SynapseModel:
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("E-prop compiler only "
                                      "supports Delta synapses")

        # Return model
        return model

    def build_weight_update_model(self, conn: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state: CompileState) -> WeightUpdateModel:
        if not is_value_constant(connect_snippet.delay):
            raise NotImplementedError("E-prop compiler only "
                                      "support heterogeneous delays")

        # Calculate membrane persistence
        alpha = np.exp(-self.dt / compile_state.tau_mem)

        # If target neuron is LIF, create weight update model with eProp LIF
        target_neuron = conn.target().neuron
        if isinstance(target_neuron, LeakyIntegrateFire):
            wum = WeightUpdateModel(
                model=eprop_lif_model,
                param_vals={"CReg": self.c_reg, "Alpha": alpha,
                            "FTarget": (self.f_target * self.dt) / 1000.0, 
                            "AlphaFAv": np.exp(-self.dt / self.tau_reg),
                            "Vthresh_post": target_neuron.v_thresh},
                var_vals={"g": connect_snippet.weight, "eFiltered": 0.0,
                          "DeltaG": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_var_vals={"Psi": 0.0, "FAvg": 0.0},
                post_neuron_var_refs={"RefracTime_post": "RefracTime",
                                      "V_post": "V", "E_post": "E"})
        # Otherise, if it's ALIF, create weight update model with eProp ALIF
        elif isinstance(target_neuron, AdaptiveLeakyIntegrateFire):
            # Calculate adaptation variable persistence
            rho = np.exp(-self.dt / compile_state.tau_adapt)

            wum = WeightUpdateModel(
                model=eprop_alif_model,
                param_vals={"CReg": self.c_reg, "Alpha": alpha, "Rho": rho, 
                            "FTarget": (self.f_target * self.dt) / 1000.0, 
                            "AlphaFAv": np.exp(-self.dt / self.tau_reg),
                            "Vthresh_post": target_neuron.v_thresh,
                            "Beta_post": target_neuron.beta},
                var_vals={"g": connect_snippet.weight, "eFiltered": 0.0,
                          "DeltaG": 0.0, "epsilonA": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_var_vals={"Psi": 0.0, "FAvg": 0.0},
                post_neuron_var_refs={"RefracTime_post": "RefracTime",
                                      "V_post": "V", "A_post": "A", 
                                      "E_post": "E"})
        # Otherwise, if target neuron is readout, create 
        # weight update model with simple output learning rule
        elif target_neuron.readout is not None:
            wum = WeightUpdateModel(
                model=output_learning_model,
                param_vals={"Alpha": alpha},
                var_vals={"g": connect_snippet.weight, "DeltaG": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_neuron_var_refs={"E_post": "E"})

            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

        # Add weights to list of checkpoint vars
        compile_state.checkpoint_connection_vars.append((conn, "g"))

        # Add connection to list of connections to optimise
        compile_state.weight_optimiser_connections.append(conn)

        # Return weight update model
        return wum

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict,
                                compile_state: CompileState) -> CompiledTrainingNetwork:
        # Fuse pre and postsynaptic updates for efficiency
        genn_model.fuse_postsynaptic_models = True
        genn_model.fuse_pre_post_weight_update_models = True

        # Correctly target feedback
        for c in compile_state.feedback_connections:
            connection_populations[c].pre_target_var = "ISynFeedback"

        # Add optimisers to connection weights that require them
        optimiser_custom_updates = []
        deep_r_record_rewirings_ccus = []
        for i, c in enumerate(compile_state.weight_optimiser_connections):
            genn_pop = connection_populations[c]

            # If connection is in list of those to use Deep-R on
            delta_g_var_ref = create_wu_var_ref(genn_pop, "DeltaG")
            weight_var_ref = create_wu_var_ref(genn_pop, "g")
            if c in self.deep_r_inh_conns or c in self.deep_r_exc_conns:
                # Add infrastructure
                excitatory = (c in self.deep_r_exc_conns)
                deep_r_2_ccu = add_deep_r(genn_pop, genn_model, self,
                                          self.deep_r_l1_strength, 
                                          delta_g_var_ref, weight_var_ref,
                                          excitatory)
                
                # If we should record rewirings from
                # this connection, add to list with key
                if c in self.deep_r_record_rewirings:
                    deep_r_record_rewirings_ccus.append(
                        (deep_r_2_ccu, self.deep_r_record_rewirings[c]))
            # Add optimiser
            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Weight{i}", weight_var_ref, delta_g_var_ref,
                    genn_model, True))

        # Add optimisers to population biases that require them
        for i, p in enumerate(compile_state.bias_optimiser_populations):
            genn_pop = neuron_populations[p]
            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Bias{i}", create_var_ref(genn_pop, "Bias"),
                    create_var_ref(genn_pop, "DeltaBias"),
                    genn_model, False))

        # Loop through populations requiring softmax
        # calculation and add requisite custom updates
        for p, o, s in compile_state.softmax_populations:
            genn_pop = neuron_populations[p]
            self.add_softmax_custom_updates(genn_model, genn_pop, 
                                            o, s)

        # Create custom updates to implement variable reset
        compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations)

        # Build list of base callbacks
        base_train_callbacks = []
        base_validate_callbacks = []
        deep_r_required = (len(self.deep_r_exc_conns) > 0 
                           or len(self.deep_r_inh_conns) > 0)

        # If Deep-R and L1 regularisation are required, add callback
        if deep_r_required and self.deep_r_l1_strength > 0.0:
            base_train_callbacks.append(CustomUpdateOnBatchEnd("DeepRL1"))

        if len(optimiser_custom_updates) > 0:
            if self.full_batch_size > 1:
                base_train_callbacks.append(CustomUpdateOnBatchEnd("GradientBatchReduce"))
            base_train_callbacks.append(CustomUpdateOnBatchEnd("GradientLearn"))
        if compile_state.is_reset_custom_update_required:
            base_train_callbacks.append(CustomUpdateOnBatchBegin("Reset"))
            base_validate_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        # If Deep-R is required, trigger Deep-R callbacks at end of batch
        if deep_r_required:
            base_train_callbacks.append(CustomUpdateOnTrainBegin("DeepRInit"))
            base_train_callbacks.append(CustomUpdateOnBatchEnd("DeepR1"))
            base_train_callbacks.append(CustomUpdateOnBatchEnd("DeepR2"))

        # Add callbacks to record number of rewirings
        for c, k in deep_r_record_rewirings_ccus:
            base_train_callbacks.append(RewiringRecord(c, k))

        # If softmax is required, add three stage reduction to callbacks
        # **NOTE** this is also required for validation because readout is configured this way
        if len(compile_state.softmax_populations) > 0:
            base_train_callbacks.append(CustomUpdateOnTimestepEnd("Softmax1"))
            base_train_callbacks.append(CustomUpdateOnTimestepEnd("Softmax2"))
            base_train_callbacks.append(CustomUpdateOnTimestepEnd("Softmax3"))
            base_validate_callbacks.append(CustomUpdateOnTimestepEnd("Softmax1"))
            base_validate_callbacks.append(CustomUpdateOnTimestepEnd("Softmax2"))
            base_validate_callbacks.append(CustomUpdateOnTimestepEnd("Softmax3"))
        
        # Build list of optimisers and their custom updates
        optimisers = []
        if len(optimiser_custom_updates) > 0:
            optimisers.append((self._optimiser, optimiser_custom_updates))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations,
            self.communicator, compile_state.losses,
            self.example_timesteps, base_train_callbacks,
            base_validate_callbacks, optimisers,
            compile_state.checkpoint_connection_vars,
            compile_state.checkpoint_population_vars, self.reset_time_between_batches)

    def _create_optimiser_custom_update(self, name_suffix, var_ref,
                                        gradient_ref, genn_model, wu):
        # If batch size is greater than 1
        if self.full_batch_size > 1:
            # Create custom update model to reduce DeltaG into a variable 
            reduction_optimiser_model = CustomUpdateModel(
                gradient_batch_reduce_model, {}, {"ReducedGradient": 0.0},
                {"Gradient": gradient_ref})

            # Add GeNN custom update to model
            genn_reduction = self.add_custom_update(
                genn_model, reduction_optimiser_model, 
                "GradientBatchReduce", "CUBatchReduce" + name_suffix)
            reduced_gradient = (create_wu_var_ref(genn_reduction,
                                                  "ReducedGradient") if wu
                                else create_var_ref(genn_reduction, 
                                                    "ReducedGradient"))
            # Create optimiser model without gradient zeroing
            # logic, connected to reduced gradient
            optimiser_model = self._optimiser.get_model(reduced_gradient,
                                                        var_ref, False, None)
        # Otherwise
        else:
            # Create optimiser model with gradient zeroing 
            # logic, connected directly to population
            optimiser_model = self._optimiser.get_model(gradient_ref, var_ref,
                                                        True, None)

        # Add GeNN custom update to model
        return self.add_custom_update(genn_model, optimiser_model,
                                      "GradientLearn",
                                      "CUGradientLearn" + name_suffix)
