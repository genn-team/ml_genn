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
from ..utils.network import get_network_dag

class CompiledFewSpikeNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations, 
                 connection_populations, k:int):
        super(CompiledFewSpikeNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations)
    
        self.k = k

    def evaluate_numpy(self, x: dict, y: dict):
        """ Evaluate an input in numpy format against labels
        accuracy --  dictionary containing accuracy of predictions 
                     made by each output Population or Layer
        """
        # Determine the number of elements in x and y
        x_size = get_numpy_size(x)
        y_size = get_numpy_size(x)
        
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
        x = batch_numpy(x, batch_size, x_size)
        y = batch_numpy(y, batch_size, y_size)
        
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
        
        for t in range(self.k):
            self.step_time()

        # Get predictions from model
        y_star = self.get_output(list(y.keys()))
        
        # Return dictionaries of output population to number of correct
        # **TODO** insert loss-function/other metric here
        return {p : np.sum((np.argmax(o_y_star[:len(o_y)], axis=1) == o_y))
                for (p, o_y), o_y_star in zip(y.items(), y_star)}


class FewSpikeCompiler(Compiler):
    def __init__(self, inputs, outputs, k: int=10, dt:float=1.0, batch_size:int=1, rng_seed:int=0,
                 kernel_profiling:bool=False, prefer_in_memory_connect=True,
                 **genn_kwargs):
        super(FewSpikeCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling, 
                                                prefer_in_memory_connect,
                                                **genn_kwargs)

        self.k = k

        # **YUCK** we want compilers to be re-usable
        self.inputs = inputs
        self.outputs = outputs
    
    def pre_compile(self, network):
        dag = get_network_dag(self.inputs,self.outputs)
        
        # **YUCK** we want compilers to be re-usable
        self.con_delay = {}
        self.pop_pipeline_depth = {}
        self.pop_alpha = {}
        
        # Loop through populations
        next_pipeline_depth = {}
        for p in dag:
            # If layer has incoming connections
            if len(p.incoming_connections) > 0:
                # Determine the maximum alpha value of presynaptic populations
                # **TODO** this should be done in the converter - neurons already check this
                max_presyn_alpha = max(c().source().neuron.alpha.value 
                                       for c in p.incoming_connections)

                # Determine the maximum pipeline depth from upstream synapses
                pipeline_depth = max(next_pipeline_depth[c().source()] 
                                       for c in p.incoming_connections)
                self.pop_pipeline_depth[p] = pipeline_depth
                
                # Downstream layer pipeline depth is one more than this
                next_pipeline_depth[p] = pipeline_depth + 1

                # Loop through incoming connections
                for c in p.incoming_connections:
                    # Set upstream delay so all spikes 
                    # arrive at correct  pipeline stage
                    source_pop = c().source()
                    depth_difference = (pipeline_depth
                                        - next_pipeline_depth[source_pop])
                    self.con_delay[c] = depth_difference * self.k

                    # Set presyn alpha to maximum alpha of all presyn layers
                    # **TODO** this should be done in the converter - neurons already check this
                    self.pop_alpha[source_pop] = max_presyn_alpha

            # Otherwise (layer is an input layer),
            # set this layer's delay as zero
            else:
                self.pop_pipeline_depth[p] = 0
                next_pipeline_depth[p] = 0

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
            custom_model = CustomUpdateModel(
                model=genn_model, param_vals={}, var_vals={}, 
                var_refs={"V": lambda nrn_pops, _: create_var_ref(nrn_pops[pop], "V"),
                          "Fx": lambda nrn_pops, _: create_var_ref(nrn_pops[pop], "Fx"),})

            # Build custom update model customised for parameters and values
            built_custom_model = build_model(custom_model)
    
            # Add to list of custom updates to be applied in "Reset" group
            custom_updates["Reset"].append(built_custom_model + (custom_model.var_refs,))
        elif not isinstance(pop.neuron, FewSpikeReluInput):
            raise NotImplementedError(
                "FewSpike models only support FewSpikeRelu "
                "and FewSpikeReluInput neurons")
        
        #if pop in self.pop_alpha:
        #     model.
        print(model)
        print(f"Pop alpha:{self.pop_alpha[pop]}")
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
                                       connection_populations, self.k)
