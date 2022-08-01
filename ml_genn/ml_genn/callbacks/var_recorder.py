import numpy as np

from typing import Sequence, Union

from ..utils.network import get_underlying_pop

Vars = Union[str, Sequence[str]]

class VarRecorder:
    def __init__(self, pop, vars: Vars):
        # Get underlying population
        # **TODO** handle Connection variables as well
        self._pop = get_underlying_pop(pop)
        self._vars = vars if isinstance(vars, Sequence) else (vars, )
        
        # Flag to track whether simualtion is batched or not
        self._in_batch = False
        
        # Create empty list to hold recorded data from each variable
        self.data = {v: [] for v in vars}
    
    def set_params(self, params):
        assert  "compiled_network" in params
        self._compiled_network = params["compiled_network"]
            
    def on_timestep_end(self):
        # Loop through variables we want to record
        for v in self.data.keys():
            # Copy variable from device
            pop = self._compiled_network.neuron_populations[self._pop]
            pop.pull_var_from_device(v)
            
            # Add copy of variable data to list
            data = np.copy(pop.vars[v].view)
            
            # If we're in a batch add data to the latest batch list
            if self._in_batch:
                self.data[v][-1].append(data)
            # Otherwise, add data directly to variable list
            else:
                self.data[v].append(data)

    def on_batch_begin(self):
        # Set flag
        self._in_batch = True
        
        # Add a new list to each variable
        for v in self.data.keys():
            self.data[v].append([])
            
            
            
