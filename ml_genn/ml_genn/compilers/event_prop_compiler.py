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
from ml_genn.synapses import Exponential
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


class CompileState:
    def __init__(self, losses, readouts):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self.weight_optimiser_connections = []
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


weight_update_model = {
    "var_name_types": [("g", "scalar", VarAccess_READ_ONLY),
                       ("DeltaG", "scalar")],

    "sim_code": """
    $(addToInSyn, $(g));
    """,
    "event_threshold_condition_code": """
    $(BackSpike_pre)
    """,
    "event_code": """
    $(DeltaG) += $(LambdaI_post);
    """}


gradient_batch_reduce_model = {
    "var_name_types": [("ReducedGradient", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    "var_refs": [("Gradient", "scalar")],
    "update_code": """
    $(ReducedGradient) = $(Gradient);
    $(Gradient) = 0;
    """}


class EventPropCompiler(Compiler):
    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 max_spikes: int = 500, dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False,
                 **genn_kwargs):
        super(EventPropCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling,
                                                prefer_in_memory_connect=True,
                                                **genn_kwargs)
        self.example_timesteps = example_timesteps
        self.losses = losses
        self.max_spikes = max_spikes
        self._optimiser = get_object(optimiser, Optimiser, "Optimiser",
                                     default_optimisers)

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
        # Otherwise, if neuron is an input
        elif not hasattr(pop.neuron, "set_input"):
            pass
        # Otherwise i.e. it's hidden
        else:
            # Add additional input variable to receive feedback
            model_copy.add_additional_input_var("RevISyn", "scalar", 0.0)

            # Add variable to hold backspike flag
            model_copy.add_var("BackSpike", "bool", False)
            
            # Add read and write indices for tape variables
            model_copy.add_var("TapeWriteIndex", "int", 0)
            model_copy.add_var("TapeReadIndex", "int", 0)
            
            # Add EGP for spike time tape variables
            model_copy.add_egp("TapeSpikeTime", "scalar*", 
                               np.empty(self.batch_size * self.max_spikes))

            # If neuron model is LIF
            if isinstance(pop.neuron, LeakyIntegrateFire):
                # Check EventProp constraints
                if pop.neuron.integrate_during_refrac:
                    logger.warning("EventProp learning works best with LIF "
                                   "neurons which do not continue to integrate "
                                   "during their refractory period")
                if pop.neuron.relative_reset:
                    logger.warning("EventProp learning works best with LIF "
                                   "neurons with an absolute reset mechanism")
                
                # Add EGP for IMinusV tape variables
                model_copy.add_egp("TapeIMinusV", "scalar*", 
                                   np.empty(self.batch_size * self.max_spikes))
                
                model_copy.prepend_sim_code(
                    f"""
                    const int bufferOffset = $(batch) * ((int)$(N_neurons))*((int) $(N_max_spike)) + $(id) * ((int) $(N_max_spike));
                    """)
                # Prepend (as it accesses the pre-reset value of V) 
                # code to write spike time and I-V to tape
                model_copy.prepend_reset_code(
                    f"""
                    if($(TapeWriteIndex) != $(fwd_start)) {{
                        // Write spike time and I-V to tape
                        $(TapeSpikeTime)[buf_idx + $(TapeWriteIndex)] = $(t);
                        $(TapeIMinusV)[buf_idx + $(TapeWriteIndex)] = $(Isyn) - $(V);
                        $(TapeWriteIndex)++;
                        
                        // Loop around if we've reached end of circular buffer
                        if ($(TapeWriteIndex) >= {self.max_spikes}) {{
                            $(TapeWriteIndex) = 0;
                        }}
                    }}
                    else {{
                        //printf("%f: hidden: ImV buffer violation in neuron %d, fwd_start: %d, new_fwd_start: %d, rp_ImV: %d, wp_ImV: %d\\n", $(t), $(id), $(fwd_start), $(new_fwd_start), $(rp_ImV), $(wp_ImV));
                        // assert(0);
                    }}
                    """)
            else:
                raise NotImplementedError(
                    f"EventProp compiler doesn't support "
                    f"{type(pop.neuron).__name__} neurons")

        # Build neuron model and return
        return model_copy

    def build_synapse_model(self, conn, model, compile_state):
        if not isinstance(conn.synapse, Exponential):
            raise NotImplementedError("EventProp compiler only "
                                      "supports Exponential synapses")

        # Return model
        # **THINK** IS scaling different?
        return model
    
    def build_weight_update_model(self, conn, weight, delay, compile_state):
        if not is_value_constant(delay):
            raise NotImplementedError("EventProp compiler only "
                                      "support heterogeneous delays")
        
        # Create basic weight update model
        wum = WeightUpdateModel(model=deepcopy(weight_update_model),
                                var_vals={"g": weight, "DeltaG": 0.0})
        
        # If source neuron is LIF, add additional event code to backpropagate gradient
        source_neuron = conn.source().neuron
        if isinstance(source_neuron, LeakyIntegrateFire):
            wum.append_event_code("$(addToPre, $(g) * ($(LambdaV_post) - $(LambdaI_post)));")

        # Add weights to list of checkpoint vars
        compile_state.checkpoint_connection_vars.append((conn, "g"))

        # Add connection to list of connections to optimise
        compile_state.weight_optimiser_connections.append(conn)

        # Return weight update model
        return wum

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations,
                                compile_state, softmax):
        # Fuse postsynaptic updates for efficiency
        genn_model._model.set_fuse_postsynaptic_models(True)

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
        
        # Create custom updates to implement variable reset
        #compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations)
        
        # Build list of base callbacks
        base_callbacks = []
        if len(optimiser_custom_updates) > 0:
            if self.batch_size > 1:
                base_callbacks.append(CustomUpdateOnBatchEnd("GradientBatchReduce"))
            base_callbacks.append(CustomUpdateOnBatchEnd("GradientLearn"))
        #if compile_state.is_reset_custom_update_required:
        #    base_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations, softmax,
            compile_state.losses, self._optimiser, self.example_timesteps,
            base_callbacks, optimiser_custom_updates,
            compile_state.checkpoint_connection_vars,
            compile_state.checkpoint_population_vars, True)

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
