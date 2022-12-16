import logging
import numpy as np

from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import (VarAccess_READ_ONLY,
                                        VarAccess_REDUCE_BATCH_SUM)
from ml_genn.callbacks import CustomUpdateOnBatchBegin, CustomUpdateOnBatchEnd
from ml_genn.compilers import Compiler
from ml_genn.compilers.compiled_training_network import CompiledTrainingNetwork
from ml_genn.callbacks import BatchProgressBar
from ml_genn.losses import Loss
from ml_genn.neurons import AdaptiveLeakyIntegrateFire, LeakyIntegrateFire
from ml_genn.optimisers import Optimiser
from ml_genn.synapses import Delta
from ml_genn.utils.callback_list import CallbackList
from ml_genn.utils.data import MetricsType
from ml_genn.utils.model import CustomUpdateModel, WeightUpdateModel

from copy import deepcopy
from pygenn.genn_model import create_var_ref, create_wu_var_ref
from ml_genn.compilers.compiler import create_reset_custom_update
from ml_genn.utils.module import get_object, get_object_mapping
from ml_genn.utils.value import is_value_constant

from ml_genn.optimisers import default_optimisers
from ml_genn.losses import default_losses

logger = logging.getLogger(__name__)

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
        self._neuron_reset_vars = {}
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []

    def add_neuron_readout_reset_vars(self, pop):
        reset_vars = pop.neuron.readout.reset_vars
        if len(reset_vars) > 0:
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
    "param_name_types": [("CReg", "scalar"), ("Alpha", "scalar"), 
                         ("FTarget", "scalar"), ("AlphaFAv", "scalar")],
    "var_name_types": [("g", "scalar", VarAccess_READ_ONLY),
                       ("eFiltered", "scalar"), ("DeltaG", "scalar")],
    "pre_var_name_types": [("ZFilter", "scalar")],
    "post_var_name_types": [("Psi", "scalar"), ("FAvg", "scalar")],

    "pre_spike_code": """
    $(ZFilter) += 1.0;
    """,
    "pre_dynamics_code": """
    $(ZFilter) *= $(Alpha);
    """,

    "post_spike_code": """
    $(FAvg) += (1.0 - $(AlphaFAv));
    """,
    "post_dynamics_code": """
    $(FAvg) *= $(AlphaFAv);
    if ($(RefracTime_post) > 0.0) {
      $(Psi) = 0.0;
    }
    else {
      $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - $(Vthresh_post)) / $(Vthresh_post)));
    }
    """,

    "sim_code": """
    $(addToInSyn, $(g));
    """,
    "synapse_dynamics_code": """
    const scalar e = $(ZFilter) * $(Psi);
    scalar eFiltered = $(eFiltered);
    eFiltered = (eFiltered * $(Alpha)) + e;
    $(DeltaG) += (eFiltered * $(E_post)) + (($(FAvg) - $(FTarget)) * $(CReg) * e);
    $(eFiltered) = eFiltered;
    """}

eprop_alif_model = {
    "param_name_types": [("CReg", "scalar"), ("Alpha", "scalar"),
                         ("Rho", "scalar"), ("FTarget", "scalar"),
                         ("AlphaFAv", "scalar")],
    "var_name_types": [("g", "scalar", VarAccess_READ_ONLY), 
                       ("eFiltered", "scalar"), ("epsilonA", "scalar"),
                       ("DeltaG", "scalar")],
    "pre_var_name_types": [("ZFilter", "scalar")],
    "post_var_name_types": [("Psi", "scalar"), ("FAvg", "scalar")],

    "pre_spike_code": """
    $(ZFilter) += 1.0;
    """,
    "pre_dynamics_code": """
    $(ZFilter) *= $(Alpha);
    """,

    "post_spike_code": """
    $(FAvg) += (1.0 - $(AlphaFAv));
    """,
    "post_dynamics_code": """
    $(FAvg) *= $(AlphaFAv);
    if ($(RefracTime_post) > 0.0) {
      $(Psi) = 0.0;
    }
    else {
      $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - ($(Vthresh_post) + ($(Beta_post) * $(A_post)))) / $(Vthresh_post)));
    }
    """,

    "sim_code": """
    $(addToInSyn, $(g));
    """,
    "synapse_dynamics_code": """
    // Calculate some common factors in e and epsilon update
    scalar epsilonA = $(epsilonA);
    const scalar psiZFilter = $(Psi) * $(ZFilter);
    const scalar psiBetaEpsilonA = $(Psi) * $(Beta_post) * epsilonA;

    // Calculate e and episilonA
    const scalar e = psiZFilter  - psiBetaEpsilonA;
    $(epsilonA) = psiZFilter + (($(Rho) * epsilonA) - psiBetaEpsilonA);

    // Calculate filtered version of eligibility trace
    scalar eFiltered = $(eFiltered);
    eFiltered = (eFiltered * $(Alpha)) + e;
    
    // Apply weight update
    $(DeltaG) += (eFiltered * $(E_post)) + (($(FAvg) - $(FTarget)) * $(CReg) * e);
    $(eFiltered) = eFiltered;
    """}

output_learning_model = {
    "param_name_types": [("Alpha", "scalar")],
    "var_name_types": [("g", "scalar", VarAccess_READ_ONLY), 
                       ("DeltaG", "scalar")],
    "pre_var_name_types": [("ZFilter", "scalar")],

    "pre_spike_code": """
    $(ZFilter) += 1.0;
    """,
    "pre_dynamics_code": """
    $(ZFilter) *= $(Alpha);
    """,

    "sim_code": """
    $(addToInSyn, $(g));
    """,
    "synapse_dynamics_code": """
    $(DeltaG) += $(ZFilter) * $(E_post);
    $(addToPre, $(g) * $(E_post));
    """}

gradient_batch_reduce_model = {
    "var_name_types": [("ReducedGradient", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    "var_refs": [("Gradient", "scalar")],
    "update_code": """
    $(ReducedGradient) = $(Gradient);
    $(Gradient) = 0;
    """}

class EPropCompiler(Compiler):
    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 tau_reg: float = 500.0, c_reg: float = 0.001, 
                 f_target: float = 10.0, train_output_bias: bool = True,
                 dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False,
                 reset_time_between_batches: bool = True, **genn_kwargs):
        super(EPropCompiler, self).__init__(dt, batch_size, rng_seed,
                                            kernel_profiling,
                                            prefer_in_memory_connect=True,
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

    def pre_compile(self, network, **kwargs):
        # Build list of output populations
        readouts = [p for p in network.populations
                    if p.neuron.readout is not None]

        return CompileState(self.losses, readouts)

    def build_neuron_model(self, pop, model, compile_state):
        # Make copy of model
        model_copy = deepcopy(model)

        # If population has a readout i.e. it's an output
        if pop.neuron.readout is not None:
            # Add any output reset variables to compile state
            compile_state.add_neuron_readout_reset_vars(pop)

            # Get loss function associated with this output neuron
            loss = compile_state.losses[pop]

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
                $(E) = $({model_copy.output_var_name}) - yTrue;
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
                model_copy.append_sim_code("$(DeltaBias) += $(E);")

                # Add population to list of those with biases to optimise
                compile_state.bias_optimiser_populations.append(pop)

                # Add bias to list of checkpoint vars
                compile_state.checkpoint_population_vars.append((pop, "Bias"))

        # Otherwise, if neuron isn't an input i.e. it's hidden
        elif not hasattr(pop.neuron, "set_input"):
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
            model_copy.append_sim_code("$(E) = $(ISynFeedback);")

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

    def build_synapse_model(self, conn, model, compile_state):
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("E-prop compiler only "
                                      "supports Delta synapses")

        # Return model
        return model
    
    def build_weight_update_model(self, conn, weight, delay, compile_state):
        if not is_value_constant(delay):
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
                            "AlphaFAv": np.exp(-self.dt / self.tau_reg)},
                var_vals={"g": weight, "eFiltered": 0.0, "DeltaG": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_var_vals={"Psi": 0.0, "FAvg": 0.0})
        # Otherise, if it's ALIF, create weight update model with eProp ALIF
        elif isinstance(target_neuron, AdaptiveLeakyIntegrateFire):
            # Calculate adaptation variable persistence
            rho = np.exp(-self.dt / compile_state.tau_adapt)
            
            wum = WeightUpdateModel(
                model=eprop_alif_model,
                param_vals={"CReg": self.c_reg, "Alpha": alpha, "Rho": rho, 
                            "FTarget": (self.f_target * self.dt) / 1000.0, 
                            "AlphaFAv": np.exp(-self.dt / self.tau_reg)},
                var_vals={"g": weight, "eFiltered": 0.0,
                          "DeltaG": 0.0, "epsilonA": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_var_vals={"Psi": 0.0, "FAvg": 0.0})
        # Otherwise, if target neuron is readout, create 
        # weight update model with simple output learning rule
        elif target_neuron.readout is not None:
            wum = WeightUpdateModel(
                model=output_learning_model,
                param_vals={"Alpha": alpha},
                var_vals={"g": weight, "DeltaG": 0.0},
                pre_var_vals={"ZFilter": 0.0})

            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

        # Add weights to list of checkpoint vars
        compile_state.checkpoint_connection_vars.append((conn, "g"))

        # Add connection to list of connections to optimise
        compile_state.weight_optimiser_connections.append(conn)

        # Return weight update model
        return wum

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations,
                                compile_state, softmax):
        # Fuse pre and postsynaptic updates for efficiency
        genn_model._model.set_fuse_postsynaptic_models(True)
        genn_model._model.set_fuse_pre_post_weight_update_models(True)

        # Correctly target feedback
        for c in compile_state.feedback_connections:
            connection_populations[c].pre_target_var = "ISynFeedback"
        
        # Add optimisers to connection weights that require them
        optimiser_custom_updates = []
        for i, c in enumerate(compile_state.weight_optimiser_connections):
            genn_pop = connection_populations[c]
            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Weight{i}", create_wu_var_ref(genn_pop, "g"),
                    create_wu_var_ref(genn_pop, "DeltaG"), genn_model))
        
        # Add optimisers to population biases that require them
        for i, p in enumerate(compile_state.bias_optimiser_populations):
            genn_pop = neuron_populations[p]
            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Bias{i}", create_var_ref(genn_pop, "Bias"),
                    create_var_ref(genn_pop, "DeltaBias"), genn_model))
        
        # Create custom updates to implement variable reset
        compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations)
        
        # Build list of base callbacks
        base_callbacks = []
        if len(optimiser_custom_updates) > 0:
            if self.batch_size > 1:
                base_callbacks.append(CustomUpdateOnBatchEnd("GradientBatchReduce"))
            base_callbacks.append(CustomUpdateOnBatchEnd("GradientLearn"))
        if compile_state.is_reset_custom_update_required:
            base_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations, softmax,
            compile_state.losses, self._optimiser, self.example_timesteps,
            base_callbacks, optimiser_custom_updates,
            compile_state.checkpoint_connection_vars,
            compile_state.checkpoint_population_vars, self.reset_time_between_batches)

    def _create_optimiser_custom_update(self, name_suffix, var_ref,
                                        gradient_ref, genn_model):
        # If batch size is greater than 1
        if self.batch_size > 1:
            # Create custom update model to reduce DeltaG into a variable 
            reduction_optimiser_model = CustomUpdateModel(
                gradient_batch_reduce_model, {}, {"ReducedGradient": 0.0},
                {"Gradient": gradient_ref})
            
            # Add GeNN custom update to model
            genn_reduction = self.add_custom_update(
                genn_model, reduction_optimiser_model, 
                "GradientBatchReduce", "CUBatchReduce" + name_suffix)
            wu = genn_reduction.custom_wu_update
            reduced_gradient = (create_wu_var_ref(genn_reduction,
                                                  "ReducedGradient") if wu
                                else create_var_ref(genn_reduction, 
                                                    "ReducedGradient"))
            # Create optimiser model without gradient zeroing
            # logic, connected to reduced gradient
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
