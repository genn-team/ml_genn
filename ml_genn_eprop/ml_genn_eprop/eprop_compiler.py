import numpy as np

from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from ml_genn.compilers import Compiler
from ml_genn.compilers.compiled_training_network import CompiledTrainingNetwork
from ml_genn.callbacks import BatchProgressBar
from ml_genn.losses import (Loss, MeanSquareError,
                            SparseCategoricalCrossentropy)
from ml_genn.neurons import LeakyIntegrateFire
from ml_genn.optimisers import Optimiser
from ml_genn.outputs import Var
from ml_genn.synapses import Delta
from ml_genn.utils.callback_list import CallbackList
from ml_genn.utils.data import MetricsType
from ml_genn.utils.model import CustomUpdateModel, WeightUpdateModel

from copy import deepcopy
from functools import partial
from pygenn.genn_model import create_var_ref, create_wu_var_ref
from ml_genn.compilers.compiler import build_model
from ml_genn.utils.module import get_object, get_object_mapping
from ml_genn.utils.value import is_value_constant

from ml_genn.optimisers import default_optimisers
from ml_genn.losses import default_losses


class CompileState:
    def __init__(self, losses, outputs):
        self.losses = get_object_mapping(losses, outputs,
                                         Loss, "Loss", default_losses)
        self._tau_mem = None
        self.feedback_connections = []
    
    def add_feedback_connection(self, conn):
        self.feedback_connections.append(conn)

    @property
    def tau_mem(self):
        assert self._tau_mem is not None

        return self._tau_mem
    
    @tau_mem.setter
    def tau_mem(self, tau_mem):
        if self._tau_mem is None:
            self._tau_mem = tau_mem
        if self._tau_mem != tau_mem:
            raise NotImplementedError("EProp compiler doesn't "
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

class EPropCompiler(Compiler):
    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 tau_reg: float = 500.0, c_reg: float = 0.001, 
                 f_target: float = 10.0, dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False, 
                 reset_time_between_batches: bool = True, **genn_kwargs):
        super(EPropCompiler, self).__init__(dt, batch_size, rng_seed,
                                            kernel_profiling,
                                            prefer_in_memory_connect=False,
                                            **genn_kwargs)
        self.example_timesteps = example_timesteps
        self.losses = losses
        self._optimiser = get_object(optimiser, Optimiser, "Optimiser",
                                     default_optimisers)
        self.tau_reg = tau_reg
        self.c_reg = c_reg
        self.f_target = f_target
        self.reset_time_between_batches = reset_time_between_batches

    def pre_compile(self, network, **kwargs):
        # Build list of output populations
        outputs = [p for p in network.populations
                   if p.neuron.output is not None]
                   
        return CompileState(self.losses, outputs)

    def build_neuron_model(self, pop, model, custom_updates,
                           pre_compile_output):
        # Make copy of model
        model_copy = deepcopy(model)

        # If population is an output
        if pop.neuron.output is not None:
            if not isinstance(pop.neuron.output, Var):
                raise NotImplementedError("EProp compiler only supports "
                                          "neurons with Var outputs")

            # Get loss function associated with this output neuron
            loss = pre_compile_output.losses[pop]

            # Add state variable to hold error
            # **NOTE** all loss functions require this!
            model_copy.add_var("E", "scalar", 0.0)

            # **TODO** bias?

            # Add loss function to neuron model
            # **THINK** semantics of this i.e. modifying inplace 
            # seem a bit different than others
            loss.add_to_neuron(model_copy, pop.shape, 
                               self.batch_size, self.example_timesteps)

            # If loss function is mean-square
            flat_shape = np.prod(pop.shape)
            if isinstance(loss, MeanSquareError):
                # Add sim-code to calculate error from difference
                # between y-star and the output variable
                out_var_name = pop.neuron.output.output_var_name
                model_copy.append_sim_code(
                    f"""
                    const unsigned int timestep = (int)round($(t) / {self.dt});
                    const unsigned int index = (timestep * {self.batch_size} * {flat_shape})
                                               + ($(batch) * {flat_shape}) + $(id);
                    $(E) = $({out_var_name}) - $(YTrue)[index];
                    """)
            # Otherwise, if it's sparse categorical
            elif isinstance(loss, SparseCategoricalCrossentropy):
                # Check shape is valid
                # **NOTE** we COULD add an elaborate mechanism to
                # create a GeNN population with next power-of-two
                # size but, once proper population reductions are
                # implemented, this issue will go away anyway
                if flat_shape not in [2, 4, 8, 16, 32]:
                    raise NotImplementedError("Currently EProp compiler only "
                                              "supports sparse categorical "
                                              "loss on output populations "
                                              "with 2, 4, 8, 16 or 32 neurons")

                # Add sim-code to copy output to a register
                out_var_name = pop.neuron.output.output_var_name
                model_copy.append_sim_code(f"scalar m = $({out_var_name});")

                # Generate sim-code to calculate max reduction
                mask = (2 ** flat_shape) - 1
                for i in range(int(np.log2(flat_shape))):
                    model_copy.append_sim_code(
                        f"m = fmax(m, __shfl_xor_sync(0x{mask:X}, m, 0x{2 ** i:X}));")

                # Add sim-code to calculate exponential
                model_copy.append_sim_code(
                    f"""
                    const scalar expPi = exp($({out_var_name}) - m);
                    scalar sumExpPi = expPi;
                    """)

                # Generate sim-code to generate second sum reduction
                for i in range(int(np.log2(flat_shape))):
                    model_copy.append_sim_code(
                        f"sumExpPi +=  __shfl_xor_sync(0x{mask:X}, sumExpPi, 0x{2 ** i:X});")

                # Add sim-code to convert label 
                # to one-hot and calculate error
                model_copy.append_sim_code(
                    f"""
                    const scalar pi = expPi / sumExpPi;
                    const scalar piStar = ($(id) == $(YTrue)[$(batch)]) ? 1.0 : 0.0;
                    $(E) = pi - piStar;
                    """)
            else:
                raise NotImplementedError("EProp compiler only supports "
                                          "MeanSquareError and "
                                          "SparseCategorical loss")
        # Otherwise, if neuron isn't an input
        elif not hasattr(pop.neuron, "set_input"):
            # Add additional input variable to receive feedback
            model_copy.add_additional_input_var("ISynFeedback", "scalar", 0.0)
            
            # Add state variable to store 
            # feedback and initialise to zero
            model_copy.add_var("E", "scalar", 0.0)
            
            # Add sim code to store incoming feedback in new state variable
            model_copy.append_sim_code("$(E) = $(ISynFeedback);")
            
            if isinstance(pop.neuron, LeakyIntegrateFire):
                # Check EProp constraints
                # **THINK** could these just be warnings?
                if not pop.neuron.integrate_during_refrac:
                    raise NotImplementedError("EProp compiler only supports "
                                              "LIF neurons which continue "
                                              "to integrate during their "
                                              "refractory period")
                if not pop.neuron.relative_reset:
                    raise NotImplementedError("EProp compiler only supports "
                                              "LIF neurons with a relative "
                                              "reset mechanism")

                # Set 
                pre_compile_output.tau_mem = pop.neuron.tau_mem
            else:
                raise NotImplementedError(f"EProp compiler doesn't support "
                                          f"{type(pop.neuron).__name__} "
                                          f"neurons")
        
        # Build neuron model and return
        return build_model(model_copy)

    def build_synapse_model(self, conn, model, custom_updates,
                            pre_compile_output):        
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("EProp compiler only "
                                      "supports Delta synapses")

        # Build synapse model and return
        return build_model(model)
    
    def build_weight_update_model(self, conn, weight, delay,
                                  custom_updates, pre_compile_output):
        if not is_value_constant(delay):
            raise NotImplementedError("EProp compiler only "
                                      "support heterogeneous delays")
        
        # Calculate decay time constants
        alpha = np.exp(-self.dt / pre_compile_output.tau_mem)
        
        # If target neuron is LIF, create weight update model with eProp
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
        # Otherwise, if target neuron is output, create 
        # weight update model with simple output learning rule
        elif target_neuron.output is not None:
            wum = WeightUpdateModel(
                model=output_learning_model,
                param_vals={"Alpha": alpha},
                var_vals={"g": weight, "DeltaG": 0.0},
                pre_var_vals={"ZFilter": 0.0})
            
            pre_compile_output.add_feedback_connection(conn)

        assert self.batch_size == 1

        # Get optimiser model for this connection
        zero_gradient = (self.batch_size == 1)
        optimiser_model = self._optimiser.get_model(
            lambda _, c_pops: create_wu_var_ref(c_pops[conn], "DeltaG"),
            lambda _, c_pops: create_wu_var_ref(c_pops[conn], "g"),
            zero_gradient)
        
        # If batch size is greater than 1, add gradient reduction
        if self.batch_size > 1:
            assert False

        # Build and add to list of custom updates
        # to be applied in "GradientLearn" group
        custom_model = build_model(optimiser_model)
        custom_updates["GradientLearn"].append(
            custom_model + (optimiser_model.var_refs,))

        # Build model and return
        model_copy, constant_param_vals, var_vals_copy, egp_vals, var_egp =\
            build_model(wum)
        return (model_copy, constant_param_vals, var_vals_copy,
                wum.pre_var_vals, wum.post_var_vals, egp_vals, var_egp)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, pre_compile_output):
        # Fuse pre and postsynaptic updates for efficiency
        genn_model._model.set_fuse_postsynaptic_models(True)
        genn_model._model.set_fuse_pre_post_weight_update_models(True)

        # Correctly target feedback
        for c in pre_compile_output.feedback_connections:
            connection_populations[c].pre_target_var = "ISynFeedback"

        return CompiledTrainingNetwork(genn_model, neuron_populations,
                                       connection_populations,
                                       pre_compile_output.losses,
                                       self._optimiser,
                                       self.example_timesteps,
                                       self.reset_time_between_batches)
