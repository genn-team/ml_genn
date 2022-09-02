import numpy as np

from collections import namedtuple
from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..losses import Loss, MeanSquareError, SparseCategoricalAccuracy
from ..neurons import LeakyIntegrateFire
from ..outputs import Var
from ..synapses import Delta
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType
from ..utils.model import CustomUpdateModel

from copy import deepcopy
from functools import partial
from pygenn.genn_model import create_var_ref, create_psm_var_ref
from .compiler import build_model
from ..utils.data import batch_dataset, get_metrics, get_dataset_size
from ..utils.module import get_object_mapping

from ..losses import default_losses

# Because we want the converter class to be reusable, we don't want
# the data to be a member, instead we encapsulate it in a tuple
PreCompileOutput = namedtuple("PreCompileOutput",
                              ["losses"])


class EPropCompiler(Compiler):
    def __init__(self, evaluate_timesteps: int, losses,
                 dt: float = 1.0, batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False, **genn_kwargs):
        super(EPropCompiler, self).__init__(dt, batch_size, rng_seed,
                                            kernel_profiling,
                                            prefer_in_memory_connect=False,
                                            **genn_kwargs)
        self.evaluate_timesteps = evaluate_timesteps
        self.losses = losses

    def pre_compile(self, network, **kwargs):
        # Build list of output populations
        outputs = [p for p in network.populations
                   if p.neuron.output is not None]
                   
        return PreCompileOutput(
            losses = get_object_mapping(self.losses, outputs,
                                        Loss, "Loss", default_losses)

    def build_neuron_model(self, pop, model, custom_updates,
                           pre_compile_output):
        # Make copy of model
        model_copy = deepcopy(model.model)

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

            # If loss function is mean-square
            if isinstance(loss, MeanSquareError):
                # Add sim-code to calculate error from difference
                # between y-star and the output variable
                out_var_name = pop.neuron.output.output_var_name
                model_copy.append_sim_code(
                    f"$(E) = $({out_var_name}) - $(YStar);")
            # Otherwise, if it's sparse categorical
            elif isinstance(loss, SparseCategoricalAccuracy):
                flat_shape = np.prod(pop.shape)
                
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
                
                
            else:
                raise NotImplementedError("EProp compiler only supports "
                                          "MeanSquareError and "
                                          "SparseCategorical loss")
        # Otherwise, if neuron isn't an input
        elif not hasattr(pop.neuron, set_input):
            # Add additional input variable to receive feedback
            model_copy.add_additional_input_var("ISynFeedback", "scalar", 0.0)
            
            # Add state variable to store 
            # feedback and initialise to zero
            model_copy.add_var("E", "scalar", 0.0)
            
            # Add sim code to store incoming feedback in new state variable
            model_copy.append_sim_code("$(E) = $(ISynFeedback);")
        
        # Build neuron model
        return super(EPropCompiler, self).build_neuron_model(
            pop, model, custom_updates, pre_compile_output)

    def build_synapse_model(self, conn, model, custom_updates,
                            pre_compile_output):        
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("EProp compiler only "
                                      "supports Delta synapses")


        return super(EPropCompiler, self).build_synapse_model(
            conn, model, custom_updates, pre_compile_output)
    
    def build_weight_update_model(self, conn, weight, delay,
                                  custom_updates, pre_compile_output):
        target_neuron = conn.target.neuron
        if isinstance(conn.target, LeakyIntegrateFire):
            if not target_neuron.integrate_during_refrac:
                raise NotImplementedError("EProp compiler only supports "
                                          "LIF neurons which continue "
                                          "to integrate during their "
                                          "refractory period")
            
            if not target.relative_reset:
                raise NotImplementedError("EProp compiler only supports "
                                          "LIF neurons with a relative "
                                          "reset mechanism")
            
        return super(EPropCompiler, self).build_weight_update_model(
            conn, weight, delay, custom_updates, pre_compile_output)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, pre_compile_output):
        return CompiledInferenceNetwork(genn_model, neuron_populations,
                                        connection_populations,
                                        self.evaluate_timesteps,
                                        self.reset_time_between_batches)
