import logging
import numpy as np

from string import Template
from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import (VarAccess_READ_ONLY,
                                        VarAccess_READ_ONLY_DUPLICATE,
                                        VarAccess_READ_ONLY_SHARED_NEURON,
                                        VarAccess_REDUCE_BATCH_SUM,
                                        VarAccess_REDUCE_NEURON_MAX,
                                        VarAccess_REDUCE_NEURON_SUM,
                                        VarAccessMode_READ_ONLY)

from .compiler import Compiler, ZeroInSyn
from .compiled_training_network import CompiledTrainingNetwork
from ..callbacks import (BatchProgressBar, Callback, CustomUpdateOnBatchBegin,
                         CustomUpdateOnBatchEnd)
from ..connection import Connection
from ..losses import Loss, SparseCategoricalCrossentropy
from ..neurons import LeakyIntegrate, LeakyIntegrateFire
from ..optimisers import Optimiser
from ..readouts import AvgVar, AvgVarExpWeight, MaxVar, SumVar
from ..synapses import Exponential
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType
from ..utils.model import CustomUpdateModel, WeightUpdateModel
from ..utils.network import PopulationType

from copy import deepcopy
from pygenn.genn_model import create_var_ref, create_wu_var_ref
from .compiler import create_reset_custom_update
from ..utils.module import get_object, get_object_mapping
from ..utils.network import get_underlying_pop
from ..utils.value import is_value_constant

from pygenn.genn_wrapper import (SynapseMatrixType_DENSE_INDIVIDUALG,
                                 SynapseMatrixType_SPARSE_INDIVIDUALG,
                                 SynapseMatrixType_PROCEDURAL_KERNELG,
                                 SynapseMatrixType_TOEPLITZ_KERNELG)
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
    def __init__(self, losses, readouts):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self.weight_optimiser_connections = []
        self._neuron_reset_vars = []
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []
        self.batch_softmax_populations = []
        self.feedback_connections = []
        self.update_trial_pops = []

    def add_neuron_reset_vars(self, pop, reset_vars, reset_ring):
        self._neuron_reset_vars.append((pop, reset_vars, reset_ring))

    def create_reset_custom_updates(self, compiler, genn_model,
                                    neuron_pops):
        # Loop through neuron variables to reset
        for i, (pop, reset_vars, reset_ring) in enumerate(self._neuron_reset_vars):
            # Create reset model
            model = create_reset_custom_update(
                reset_vars,
                lambda name: create_var_ref(neuron_pops[pop], name))
            
            if reset_ring:
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
                    $(RingWriteStartOffset) = $(RingReadOffset);
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
        # Set extra global parameter to batch ID
        # **TODO** this should be modifiable parameter in GeNN 5
        self.genn_pop.extra_global_params["Trial"].view[:] = batch


weight_update_model = {
    "param_name_types": [("TauSyn", "scalar")],
    "var_name_types": [("g", "scalar", VarAccess_READ_ONLY),
                       ("Gradient", "scalar")],

    "sim_code": """
    $(addToInSyn, $(g));
    """,
    "event_threshold_condition_code": """
    $(BackSpike_pre)
    """,
    "event_code": """
    $(Gradient) -= ($(LambdaI_post) * $(TauSyn));
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
    const scalar backT = $example_time - $$(t) - DT;

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
    if ($$(RingReadOffset) != $$(RingReadEndOffset) && fabs(backT - $$(RingSpikeTime)[ringOffset + $$(RingReadOffset)] - DT) < 1e-3*DT) {
        $$(BackSpike) = true;
    }

    // Forward pass
    """)

# Template used to generate reset code for neurons
neuron_reset = Template(
    """
    if($$(RingWriteOffset) != $$(RingReadEndOffset)) {
        // Write spike time and I-V to tape
        $$(RingSpikeTime)[ringOffset + $$(RingWriteOffset)] = $$(t);
        $write
        $$(RingWriteOffset)++;
        
        // Loop around if we've reached end of circular buffer
        if ($$(RingWriteOffset) >= $max_spikes) {
            $$(RingWriteOffset) = 0;
        }
    }
    else {
        //printf("%f: hidden: ImV buffer violation in neuron %d, fwd_start: %d, new_fwd_start: %d, rp_ImV: %d, wp_ImV: %d\\n", $$(t), $$(id), $$(RingReadEndOffset), $$(RingWriteStartOffset), $$(RingReadOffset), $$(RingWriteOffset));
        // assert(0);
    }
    """)

class EventPropCompiler(Compiler):
    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 reg_lambda_upper: float = 0.0, reg_lambda_lower: float = 0.0,
                 reg_nu_upper: float = 0.0, max_spikes: int = 500, 
                 dt: float = 1.0, batch_size: int = 1, rng_seed: int = 0, 
                 kernel_profiling: bool = False, **genn_kwargs):
        supported_matrix_types = [SynapseMatrixType_TOEPLITZ_KERNELG,
                                  SynapseMatrixType_PROCEDURAL_KERNELG,
                                  SynapseMatrixType_DENSE_INDIVIDUALG,
                                  SynapseMatrixType_SPARSE_INDIVIDUALG]
        super(EventPropCompiler, self).__init__(supported_matrix_types, dt,
                                                batch_size, rng_seed,
                                                kernel_profiling,
                                                **genn_kwargs)
        self.example_timesteps = example_timesteps
        self.losses = losses
        self.max_spikes = max_spikes
        self.reg_lambda_upper = reg_lambda_upper
        self.reg_lambda_lower = reg_lambda_lower
        self.reg_nu_upper = reg_nu_upper
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
            # Check loss function is compatible
            if not isinstance(compile_state.losses[pop], 
                              SparseCategoricalCrossentropy):
                raise NotImplementedError(
                    f"EventProp compiler doesn't support "
                    f"{type(loss).__name__} loss")

            # Add output logic to model
            model_copy = pop.neuron.readout.add_readout_logic(
                model_copy, max_time_required=True, dt=self.dt,
                example_timesteps=self.example_timesteps)

            # Add variable, shared across neurons to hold true label for batch
            # **HACK** we don't want to call add_to_neuron on loss function as
            # it will add unwanted code to end of neuron but we do want this
            model_copy.add_var("YTrue", "uint8_t", 0, 
                               VarAccess_READ_ONLY_SHARED_NEURON)
            
            # Add second variable to hold the true label for the backward pass
            model_copy.add_var("YTrueBack", "uint8_t", 0, 
                               VarAccess_READ_ONLY_SHARED_NEURON)

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
                                     tau_mem / (tau_syn - tau_mem))

                # Add extra global parameter to contain trial index and add 
                # population to list of those which require it updating
                model_copy.add_egp("Trial", "unsigned int", 0)
                compile_state.update_trial_pops.append(pop)

                # Add state variable to hold softmax of output
                model_copy.add_var("Softmax", "scalar", 0.0,
                                   VarAccess_READ_ONLY_DUPLICATE)

                # If readout is AvgVar or SumVar
                if isinstance(pop.neuron.readout, (AvgVar, SumVar)):
                    model_copy.prepend_sim_code(
                        f"""
                        if ($(Trial) > 0) {{
                            const scalar g = ($(id) == $(YTrueBack)) ? (1.0 - $(Softmax)) : -$(Softmax);
                            $(LambdaV) += (g / ($(TauM) * $(num_batch) * {self.dt * self.example_timesteps})) * DT; // simple Euler
                        }}
                        
                        // Forward pass
                        """)

                    # Add custom updates to calculate softmax from VSum or VAvg
                    var = ("VSum" if isinstance(pop.neuron.readout, SumVar)
                           else "VAvg")
                    compile_state.batch_softmax_populations.append(
                        (pop, var, "Softmax"))

                    # Add custom update to reset state
                    compile_state.add_neuron_reset_vars(
                        pop, model_copy.reset_vars, False)
                # Otherwise, if readout is AvgVarExpWeight
                elif isinstance(pop.neuron.readout, AvgVarExpWeight):
                    local_t_scale = 1.0 / (self.dt * self.example_timesteps)
                    model_copy.prepend_sim_code(
                        f"""
                        if ($(Trial) > 0) {{
                            const scalar g = ($(id) == $(YTrueBack)) ? (1.0 - $(Softmax)) : -$(Softmax);
                            $(LambdaV) += ((g * exp(-(1.0 - ($(t) * {local_t_scale})))) / ($(TauM) * $(num_batch) * {self.dt * self.example_timesteps})) * DT; // simple Euler
                        }}
                        
                        // Forward pass
                        """)

                    # Add custom updates to calculate softmax from VAvg
                    compile_state.batch_softmax_populations.append(
                        (pop, "VAvg", "Softmax"))

                    # Add custom update to reset state
                    compile_state.add_neuron_reset_vars(
                        pop, model_copy.reset_vars, False)
                # Otherwise, if readout is MaxVar
                elif isinstance(pop.neuron.readout, MaxVar):
                    # Add state variable to hold vmax from previous trial
                    model_copy.add_var("VMaxTimeBack", "scalar", 0.0,
                                       VarAccess_READ_ONLY_DUPLICATE)
                                       
                    model_copy.prepend_sim_code(
                        f"""
                        if ($(Trial) > 0 && fabs(backT - $(VMaxTimeBack)) < 1e-3*DT) {{
                            const scalar g = ($(id) == $(YTrueBack)) ? (1.0 - $(Softmax)) : -$(Softmax);
                            $(LambdaV) += (g / ($(TauM) * $(num_batch) * {self.dt * self.example_timesteps})); // simple Euler
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
                        False)
                # Otherwise, unsupported readout type
                else:
                    raise NotImplementedError(
                        f"EventProp compiler doesn't support "
                        f"{type(pop.neuron.readout).__name__} readouts")

                # Prepend standard code to update LambdaV and LambdaI
                model_copy.prepend_sim_code(
                    f"""
                    const float backT = {self.example_timesteps * self.dt} - $(t) - DT;

                    // Backward pass
                    $(LambdaI) = ($(A) * $(LambdaV) * ($(Beta) - $(Alpha))) + ($(LambdaI) * $(Beta));
                    $(LambdaV) *= $(Alpha);
                    """)

                # Add second reset custom update to reset YTrueBack to YTrue
                # **NOTE** seperate as these are SHARED_NEURON variables
                compile_state.add_neuron_reset_vars(
                    pop, [("YTrueBack", "uint8_t", "YTrue")], False)
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
            if hasattr(pop.neuron, "set_input"):
                # Add reset logic to reset any state 
                # variables from the original model
                compile_state.add_neuron_reset_vars(pop, model.reset_vars,
                                                    True)

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
                        write=""))
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
                        $(LambdaV) += (1.0 / $(RingIMinusV)[ringOffset + $(RingReadOffset)]) * ($(Vthresh) * $(LambdaV) + $(RevISyn));
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
                                           VarAccess_READ_ONLY_DUPLICATE)

                        # Add parameters for regulariser
                        model_copy.add_param("RegNuUpper", "int",
                                             self.reg_nu_upper)
                        model_copy.add_param("RegLambdaUpper", "int",
                                             self.reg_lambda_upper / self.batch_size)
                        model_copy.add_param("RegLambdaLower", "int",
                                             self.reg_lambda_lower / self.batch_size)

                        # Add reset variables to copy SpikeCount
                        # into SpikeCountBack and zero SpikeCount
                        additional_reset_vars.extend(
                            [("SpikeCountBack", "int", "SpikeCount"),
                             ("SpikeCount", "int", 0)])
                        
                        # Add additional transition code to apply regularisation
                        transition_code += """
                        if ($(SpikeCountBack) > $(RegNuUpper)) {
                            $(LambdaV) -= $(RegLambdaUpper) * ($(SpikeCountBack) - $(RegNuUpper));
                        }
                        else {
                            $(LambdaV) -= $(RegLambdaLower) * ($(SpikeCountBack) - $(RegNuUpper));
                        }
                        """

                        # Add code to update SpikeCount in forward reset code
                        model_copy.append_reset_code("$(SpikeCount)++;")

                    # Add reset logic to reset adjoint state variables 
                    # as well as any state variables from the original model
                    compile_state.add_neuron_reset_vars(
                        pop, model.reset_vars + additional_reset_vars,
                        True)

                    # Add code to start of sim code to run backwards pass 
                    # and handle back spikes with correct LIF dynamics
                    model_copy.prepend_sim_code(
                        neuron_backward_pass.substitute(
                            max_spikes=self.max_spikes,
                            example_time=(self.example_timesteps * self.dt),
                            dynamics="""
                            $(LambdaI) = ($(A) * $(LambdaV) * ($(Beta) - $(Alpha))) + ($(LambdaI) * $(Beta));
                            $(LambdaV) *= $(Alpha);
                            """,
                            transition=transition_code))

                    # Prepend (as it accesses the pre-reset value of V) 
                    # code to reset to write spike time and I-V to ring buffer
                    model_copy.prepend_reset_code(
                        neuron_reset.substitute(
                            max_spikes=self.max_spikes,
                            write="$(RingIMinusV)[ringOffset + $(RingWriteOffset)] = $(Isyn) - $(V);"))
                # Otherwise, neuron type is unsupported
                else:
                    raise NotImplementedError(
                        f"EventProp compiler doesn't support "
                        f"{type(pop.neuron).__name__} neurons")

        # Build neuron model and return
        return model_copy

    def build_synapse_model(self, conn, model, compile_state):
        # **NOTE** this is probably not necessary as 
        # it's also checked in build_neuron_model
        if not isinstance(conn.synapse, Exponential):
            raise NotImplementedError("EventProp compiler only "
                                      "supports Exponential synapses")

        # Return model
        # **THINK** IS scaling different?
        return model
    
    def build_weight_update_model(self, conn, connect_snippet, compile_state):
        if not is_value_constant(connect_snippet.delay):
            raise NotImplementedError("EventProp compiler only "
                                      "support heterogeneous delays")
        
        # **NOTE** this is probably not necessary as 
        # it's also checked in build_neuron_model
        if isinstance(conn.synapse, Exponential):
            tau_syn = conn.synapse.tau
        else:
            raise NotImplementedError("EventProp compiler only "
                                      "supports Exponential synapses")
                              
        # Create basic weight update model
        wum = WeightUpdateModel(model=deepcopy(weight_update_model),
                                param_vals={"TauSyn": tau_syn},
                                var_vals={"g": connect_snippet.weight,
                                          "Gradient": 0.0})
        
        # If source neuron isn't an input neuron
        source_neuron = conn.source().neuron
        if not hasattr(source_neuron, "set_input"):
            # Add connection to list of feedback connections
            compile_state.feedback_connections.append(conn)

            # If it's LIF, add additional event code to backpropagate gradient
            if isinstance(source_neuron, LeakyIntegrateFire):
                wum.append_event_code("$(addToPre, $(g) * ($(LambdaV_post) - $(LambdaI_post)));")

        # Add weights to list of checkpoint vars
        compile_state.checkpoint_connection_vars.append((conn, "g"))

        # Add connection to list of connections to optimise
        compile_state.weight_optimiser_connections.append(conn)

        # Return weight update model
        return wum

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, compile_state):
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
        
        # Add softmax custom updates for each population that requires them
        for i, (p, i, o) in enumerate(compile_state.batch_softmax_populations):
            genn_pop = neuron_populations[p]
            self.add_softmax_custom_updates(genn_model, genn_pop,
                                            i, o, p.name, "Batch")

        # Create custom updates to implement variable reset
        compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations)
        
        # Build list of base callbacks
        base_callbacks = []
        if len(optimiser_custom_updates) > 0:
            if self.batch_size > 1:
                base_callbacks.append(
                    CustomUpdateOnBatchEnd("GradientBatchReduce"))
            base_callbacks.append(
                CustomUpdateOnBatchEnd("GradientLearn"))
        
        # Add callbacks to set Trial extra global parameter 
        # on populations which require it
        for p in compile_state.update_trial_pops:
            base_callbacks.append(UpdateTrial(neuron_populations[p]))

        # Add callbacks to zero insyn on all connections
        # **NOTE** it would be great to be able to do this on device
        for genn_syn_pop in connection_populations.values():
            base_callbacks.append(
                ZeroInSyn(genn_syn_pop, self.example_timesteps))
    
        # If softmax calculation is required at end of batch, add callbacks
        if len(compile_state.batch_softmax_populations) > 0:
            base_callbacks.append(CustomUpdateOnBatchEnd("BatchSoftmax1"))
            base_callbacks.append(CustomUpdateOnBatchEnd("BatchSoftmax2"))
            base_callbacks.append(CustomUpdateOnBatchEnd("BatchSoftmax3"))

        # Add reset custom updates
        if compile_state.is_reset_custom_update_required:
            base_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations,
            compile_state.losses, self._optimiser, self.example_timesteps,
            base_callbacks, optimiser_custom_updates,
            compile_state.checkpoint_connection_vars,
            compile_state.checkpoint_population_vars, True)

    @property
    def regulariser_enabled(self):
        return (self.reg_lambda_lower != 0.0 
                or self.reg_lambda_upper != 0.0)

    def _create_optimiser_custom_update(self, name_suffix, var_ref,
                                        gradient_ref, genn_model):
        # If batch size is greater than 1
        if self.batch_size > 1:
            # Create custom update model to reduce Gradient into a variable 
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
