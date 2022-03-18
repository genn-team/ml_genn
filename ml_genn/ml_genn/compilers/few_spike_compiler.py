import numpy as np

from typing import Sequence, Union
from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..neurons import FewSpikeRelu, FewSpikeReluInput
from ..synapses import Delta
from ..utils import CustomUpdateModel
from ..utils.data import batch_numpy, get_numpy_size

from pygenn.genn_model import create_var_ref
from .compiler import build_model

class CompiledFewSpikeNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations, 
                 connection_populations, evaluate_timesteps:int):
        super(CompiledFewSpikeNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations)
    
        self.evaluate_timesteps = evaluate_timesteps

    def evaluate_numpy(self, x: dict, y: dict):
        """ Evaluate an input in numpy format against labels
        accuracy --  dictionary containing accuracy of predictions 
                     made by each output Population or Layer
        """
        # Determine the number of elements in x and y
        x_size = _get_numpy_size(x)
        y_size = _get_numpy_size(x)
        
        if x_size is None:
            raise RuntimeError("Each input population must be "
                               " provided with same number of inputs")
        if y_size is None:
            raise RuntimeError("Each output population must be "
                               " provided with same number of labels")
        if x_size != y_size:
            raise RuntimeError("Number of inputs and labels must match")
        
        total_correct = {p: 0 for p in y.keys()}
        
        # Batch x and y
        batch_size = self.genn_model.batch_size
        x = _batch_numpy(x, batch_size, x_size)
        y = _batch_numpy(y, batch_size, y_size)
        
        # Loop through batches
        for x_batch, y_batch in zip(x, y):
            # Get predictions from batch
            correct = self.evaluate_batch(x_batch, y_batch)
            
            # Add to total correct
            for p, c in correct.items():
                total_correct[p] += c
        
        # Return dictionary containing correct count
        return {p: c / x_size for p, c in total_correct.items()}
    
    def evaluate_batch(self, x: dict, y: dict):
        """ Evaluate a single batch of inputs against labels
        Args:
        x --        dict mapping input Population or InputLayer to 
                    array containing one batch of inputs
        y --        dict mapping output Population or Layer to
                    array containing one batch of labels

        Returns:
        correct --  dictionary containing number of correct predictions 
                    made by each output Population or Layer
        """
        self.custom_update("Reset")

        # Apply inputs to model
        self.set_input(x)
        
        for t in range(self.evaluate_timesteps):
            self.step_time()

        # Get predictions from model
        y_star = self.get_output(list(y.keys()))
        
        # Return dictionaries of output population to number of correct
        # **TODO** insert loss-function/other metric here
        return {p : np.sum((np.argmax(o_y_star[:len(o_y)], axis=1) == o_y))
                for (p, o_y), o_y_star in zip(y.items(), y_star)}

"""
def pre_compile(self, mlg_network, mlg_network_inputs, mlg_model_outputs):
    # Get DAG of network
    dag = get_network_dag(mlg_network_inputs, mlg_model_outputs)
    
    # Loop through populations
    next_pipeline_depth = {}
    for p in dag:
        # If layer has incoming connections
        if len(p.incoming_connections) > 0:

            # Determine the maximum alpha value of presynaptic populations
            max_presyn_alpha = max(c.source().neuron.alpha.value 
                                   for c in p.incoming_connections)

            # Determine the maximum pipeline depth from upstream synapses
            l.pipeline_depth = max(next_pipeline_depth[c.source()] 
                                   for c in p.incoming_connections)

            # Downstream layer pipeline depth is one more than this
            next_pipeline_depth[l] = l.pipeline_depth + 1

            # Loop through upstream synapses
            for s in l.upstream_synapses:
                # Set upstream delay so all spikes arrive at correct pipeline stage
                depth_difference = l.pipeline_depth - next_pipeline_depth[s.source()]
                s.delay = depth_difference * self.K

                # Set presyn alpha to maximum alpha of all presyn layers
                s.source().neuron.alpha.value = max_presyn_alpha

        # Otherwise (layer is an input layer), set this layer's delay as zero
        else:
            l.pipeline_depth = 0
            next_pipeline_depth[l] = 0
"""
class FewSpikeCompiler(Compiler):
    def __init__(self, evaluate_timesteps:int, dt:float=1.0, batch_size:int=1, rng_seed:int=0,
                 kernel_profiling:bool=False, prefer_in_memory_connect=True,
                 **genn_kwargs):
        super(FewSpikeCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling, 
                                                prefer_in_memory_connect,
                                                **genn_kwargs)
        self.evaluate_timesteps = evaluate_timesteps

    def build_neuron_model(self, pop, model, custom_updates):
        if isinstance(pop.neuron, FewSpikeRelu):
            # Define custom model for resetting
            genn_model = {
                "var_refs": [("Fx", "scalar"), ("V", "scalar")],
                "update_code":
                    """
                    $(V) = $(Fx);
                    $(Fx) = 0.0;
                    """}
            
            # Create empty model
            model = CustomUpdateModel(
                model=genn_model, param_vals={}, var_vals={}, 
                var_refs={"V": lambda nrn_pops, _, name: create_var_ref(nrn_pops[pop], "V"),
                          "Fx": lambda nrn_pops, _, name: create_var_ref(nrn_pops[pop], "Fx"),})

            # Build custom update model customised for parameters and values
            custom_model = build_model(model)
    
            # Add to list of custom updates to be applied in "Reset" group
            custom_updates["Reset"].append(custom_model + (model.var_refs,))
        else if not isinstance(pop.neuron, FewSpikeReluInput):
            raise NotImplementedError(
                "FewSpike models only support FewSpikeRelu "
                "and FewSpikeReluInput neurons")
      
        # Build neuron model
        return super(FewSpikeCompiler, self).build_neuron_model(
            pop, model, custom_updates)

    def build_synapse_model(self, conn, model, custom_updates):
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("FewSpike models only support Delta synapses")

        return super(FewSpikeCompiler, self).build_synapse_model(
            conn, model, custom_updates)
    
    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations):
        return CompiledFewSpikeNetwork(genn_model, neuron_populations, 
                                       connection_populations,
                                       self.evaluate_timesteps)
