import logging
import numpy as np
import tensorflow as tf

from collections import namedtuple
from pygenn import CustomUpdateVarAccess, VarAccessMode
from ml_genn.callbacks import CustomUpdateOnTimestepEnd
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import (BinarySpikeInput, IntegrateFire,
                             IntegrateFireInput, PoissonInput)
from ml_genn.utils.model import CustomUpdateModel
from .converter import Converter
from .enum import InputType

from copy import deepcopy
from pygenn import create_var_ref
from ml_genn.utils.network import get_network_dag

logger = logging.getLogger(__name__)

# First pass of threshold update - calculate max across batches and zero
threshold_1_model = {
    "vars": [("MaxV", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_MAX),
             ("Vthresh", "scalar", CustomUpdateVarAccess.READ_ONLY_SHARED_NEURON)],
    "var_refs": [("V", "scalar")],
    "update_code": """
    MaxV = fmax(V, Vthresh);
    V = 0.0;
    """}

# Second pass of threshold update - calculate max across neurons
threshold_2_model = {
    "var_refs": [("MaxV", "scalar", VarAccessMode.READ_ONLY),
                 ("Vthresh", "scalar", VarAccessMode.REDUCE_MAX)],
    "update_code": """
    Vthresh = MaxV;
    """}

class NormCompiler(InferenceCompiler):
    def build_neuron_model(self, pop, model, compile_state):
        # If neuron is an integrate and fire model i.e. not an input layer
        if isinstance(pop.neuron, IntegrateFire):
            # Make a copy of model with threshold
            # parameter implemented as EGP 
            model = deepcopy(model)
            model.set_param_dynamic("Vthresh")

        # Build neuron model
        return super(NormCompiler, self).build_neuron_model(
            pop, model, compile_state)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, compile_state):
        # Loop through model populations
        pop_threshold_custom_updates = {}
        for pop, genn_pop in neuron_populations.items():
            if isinstance(pop.neuron, IntegrateFire):
                # Create custom update model to implement 
                # first threshold calculation pass and add to model
                threshold_1 = CustomUpdateModel(
                    threshold_1_model, {}, 
                    {"MaxV": 0.0, "Vthresh": 0.0}, 
                    {"V": create_var_ref(genn_pop, "V")})
                genn_threshold_1 = self.add_custom_update(
                    genn_model, threshold_1, 
                    "UpdateThresh1" + pop.name, "CUThreshold1" + pop.name)
                
                # Create custom update model to implement 
                # second threshold calculation pass and add to model
                threshold_2 = CustomUpdateModel(
                    threshold_2_model, {}, {}, 
                    {"MaxV": create_var_ref(genn_threshold_1, "MaxV"),
                     "Vthresh": create_var_ref(genn_threshold_1, "Vthresh")})
                genn_threshold_2 = self.add_custom_update(
                    genn_model, threshold_2, 
                    "UpdateThresh2" + pop.name, "CUThreshold2" + pop.name)
                 
                pop_threshold_custom_updates[pop] = genn_threshold_1
        # Superclass
        compiled_net = super(NormCompiler, self).create_compiled_network(
            genn_model, neuron_populations, connection_populations, 
            compile_state)
        
        # **YUCK** monkey patch compiled network with dictionary of custom 
        # updates responsible for calculating each population's thresholds
        compiled_net.pop_threshold_custom_updates =\
            pop_threshold_custom_updates
        return compiled_net

def spike_normalise(net, net_inputs, net_outputs, norm_data,
                    evaluate_timesteps: int, num_batches: int = None,
                    dt: float = 1.0, batch_size: int = 1,
                    rng_seed: int = 0, kernel_profiling: bool = False,
                    prefer_in_memory_connect=True,
                    reset_time_between_batches=True, **genn_kwargs):
    # Don't allow models with multiple input layers
    if len(net_inputs) != 1:
        raise NotImplementedError("Spike norm does not support "
                                  "models with multiple input layers")
    
    # Create normalisation compiler
    compiler = NormCompiler(evaluate_timesteps, dt, batch_size, rng_seed,
                            kernel_profiling, prefer_in_memory_connect,
                            reset_time_between_batches, **genn_kwargs)
    
    # Use it to compile network
    compiled_net = compiler.compile(net, inputs=net_inputs, outputs=net_outputs)
    
    # Topologically sort model 
    dag = get_network_dag(net_inputs, net_outputs)

    # As we might give up optimising before the end,
    # set all population's 'final' thresholds to 1
    final_thresholds = {p: 1.0 
                        for p in dag 
                        if isinstance(p.neuron, IntegrateFire)}

    # Load compiled network
    with compiled_net:
        # Loop through populations
        for pop in dag:
            # Break at branch (many outbound)
            if len(pop.outgoing_connections) > 1:
                break

            # Skip any non-integrate and fire layers
            if not isinstance(pop.neuron, IntegrateFire):
                continue

            # Evaluate normalisation data triggering the custom updates to
            # calculate the threshold for this layer after each timestep
            # **YUCK** it would be kinda nice to turn off metrics here
            callbacks=["batch_progress_bar", 
                       CustomUpdateOnTimestepEnd("UpdateThresh1" + pop.name),
                       CustomUpdateOnTimestepEnd("UpdateThresh2" + pop.name)]
            compiled_net.evaluate_batch_iter(net_inputs, net_outputs,
                                             iter(norm_data),
                                             num_batches=num_batches,
                                             callbacks=callbacks)
            
            # Get threshold calculated for this layer across whole dataset
            threshold_cu = compiled_net.pop_threshold_custom_updates[pop]
            threshold_cu.vars["Vthresh"].pull_from_device()
            final_thresh = threshold_cu.vars["Vthresh"].view[0][0]
            
            # Set this threshold in this population's EGP
            genn_pop = compiled_net.neuron_populations[pop]
            genn_pop.extra_global_params["Vthresh"].view[:] = final_thresh

            # Copy it into final threshold dictionary
            final_thresholds[pop] = final_thresh
