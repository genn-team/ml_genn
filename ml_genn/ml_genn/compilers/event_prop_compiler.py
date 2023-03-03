import logging
import numpy as np

from string import Template
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

class CompileState:
    def __init__(self, losses, readouts):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self.weight_optimiser_connections = []
        self._neuron_reset_vars = {}
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []

    def add_neuron_reset_vars(self, pop, reset_vars):
        reset_vars += pop.neuron.readout.reset_vars
        self._neuron_reset_vars[pop] = reset_vars

    def create_reset_custom_updates(self, compiler, genn_model,
                                    neuron_pops):
        # Loop through neuron variables to reset
        for i, (pop, reset_vars) in enumerate(self._neuron_reset_vars.items()):
            # Create reset model
            model = create_reset_custom_update(
                reset_vars,
                lambda name: create_var_ref(neuron_pops[pop], name))
            
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
                $(RingReadOffset) = $(RingWriteOffset) - 1;
                if ($(RingReadOffset) < 0) {{
                    $(RingReadOffset) = {compiler.max_spikes - 1};
                }}
                $(RingReadEndOffset) = $(RingWriteStartOffset);
                $(RingWriteStartOffset) = $(RingReadOffset)
                """)
            
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

# Template used to generate backward passes for neurons
neuron_backward_pass = Template(
    """
    const int ringOffset = ($$(batch) * $$(num) * $max_spikes) + ($$(id) * $max_spikes);
    const int backT = $example_time - $$(t) - DT;
    
    // Backward pass
    $dynamics
    if ($$(BackSpike)) {
        $transition
        
        // Decrease read pointer
        $$(RingReadOffset)--;
        if ($$(RingReadOffset) < 0) {
            $$(RingReadOffset) = $max_spikes - 1;
        }
        $$(BackSpike) = false;
    }
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    if (fabs(backT - $$(RingSpikeTime)[ringOffset + $$(RingReadOffset)] - DT) < 1e-3*DT) {
        $$(BackSpike) = true;
    }

    // Forward pass
    """)

# Template used to generate reset code for neurons
neuron_reset = Template(
    """
    if($$(RingWriteOffset) != $$(RingReadEndOffset)) {
        // Write spike time and I-V to tape
        $$(RingSpikeTime)[ringOffset + $$(RingWriteOffset)] = $(t);
        $write
        $$(RingWriteIndex)++;
        
        // Loop around if we've reached end of circular buffer
        if ($$(RingWriteOffset) >= $max_spikes) {{
            $$(RingWriteOffset) = 0;
        }
    }
    else {
        //printf("%f: hidden: ImV buffer violation in neuron %d, fwd_start: %d, new_fwd_start: %d, rp_ImV: %d, wp_ImV: %d\\n", $(t), $(id), $(fwd_start), $(new_fwd_start), $(rp_ImV), $(wp_ImV));
        // assert(0);
    }
    """)

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
            compile_state.add_neuron_reset_vars(pop)

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
        # Otherwise, it's either an input or a hidden neuron
        else:
            # Add variables to hold offsets for reading and writing ring variables
            model_copy.add_var("RingWriteOffset", "int", 0)
            model_copy.add_var("RingReadOffset", "int", 0)
            
            # Add variables to hold offsets where this neuron
            # started writing to ring during the forward
            # pass and where data to read during backward pass ends
            model_copy.add_var("RingWriteStartOffset", "int", 0)
            model_copy.add_var("RingReadEndOffset", "int", 0)
            
            # Add variable to hold backspike flag
            model_copy.add_var("BackSpike", "bool", False)
                
            # Add EGP for spike time ring variables
            ring_size = self.batch_size * np.prod(pop.shape) * self.max_spikes
            model_copy.add_egp("RingSpikeTime", "scalar*", np.empty(ring_size))
            
            # If neuron is an input
            if hasattr(pop.neuron, "set_input"):
                 # Add reset logic to reset any state 
                 # variables from the original model
                compile_state.add_neuron_reset_vars(pop, model.reset_vars)

                # Add code to start of sim code to run 
                # backwards pass and handle back spikes
                model_copy.prepend_sim_code(
                    neuron_backward_pass.substitute(
                        max_spikes=self.max_spikes,
                        example_time=(self.example_timesteps * self.DT),
                        dynamics="",
                        transition=""))
                
                # Prepend code to reset to write spike time to ring buffer
                model_copy.prepend_reset_code(
                    neuron_reset.substitute(
                        max_spikes=self.max_spikes,
                        write=""))
            # Otherwise i.e. it's hidden
            else:
                # Add additional input variable to receive feedback
                model_copy.add_additional_input_var("RevISyn", "scalar", 0.0)
                
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
                    
                    # Add adjoint state variables
                    model_copy.add_var("LambdaV", "scalar", 0.0)
                    model_copy.add_var("LambdaI", "scalar", 0.0)
                    
                    # Add EGP for IMinusV ring variables
                    model_copy.add_egp("RingIMinusV", "scalar*", np.empty(ring_size))
                    
                    # Add reset logic to reset adjoint state variables 
                    # as well as any state variables from the original model
                    compile_state.add_neuron_reset_vars(
                        pop, model.reset_vars + [("LambdaV", "scalar", 0.0),
                                                 ("LambdaI", "scalar", 0.0)])
                
                    # Add code to start of sim code to run backwards pass 
                    # and handle back spikes with correct LIF dynamics
                    model_copy.prepend_sim_code(
                        neuron_backward_pass.substitute(
                            max_spikes=self.max_spikes,
                            example_time=(self.example_timesteps * self.DT),
                            dynamics="""
                            $(LambdaI) = $(tau_m) / ($(tau_syn) - $(tau_m)) * $(LambdaV) * (exp(-DT/$(tau_syn)) - $(Alpha)) + $(LambdaI) * exp(-DT/$(tau_syn));
                            $(LambdaV) *= $(Alpha);
                            """,
                            transition="$(LambdaV) += ((1.0 / $(RingIMinusV)[ringOffset + $(RingReadOffset)]) * $(Vthresh) * $(LambdaV)) + $(RevISyn);"))
                    
                    # Prepend (as it accesses the pre-reset value of V) 
                    # code to reset t[ write spike time and I-V to ring buffer
                    model_copy.prepend_reset_code(
                        neuron_reset.substitute(
                            max_spikes=self.max_spikes,
                            write="$(RingIMinusV)[ringOffset + $(RingWriteOffset)] = $(Isyn) - $(V);"))
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
