import numpy as np

from typing import Sequence, Union

from ..utils.network import get_underlying_pop

class VarRecorder:
    def __init__(self, pop, var: str):
        # Get underlying population
        # **TODO** handle Connection variables as well
        self._pop = get_underlying_pop(pop)
        self._var = var
        
        # Flag to track whether simualtion is batched or not
        self._in_batch = False
        
        # Create empty list to hold recorded data
        self.data = []
    
    def set_params(self, compiled_network, **kwargs):
        self._compiled_network = compiled_network
            
    def on_timestep_end(self, timestep):
        # Copy variable from device
        pop = self._compiled_network.neuron_populations[self._pop]
        pop.pull_var_from_device(self._var)
        
        # Add copy of variable data to list
        data = np.copy(pop.vars[self._var].view)
        
        # If we're in a batch add data to the latest batch list
        if self._in_batch:
            self.data[-1].append(data)
        # Otherwise, add data directly to variable list
        else:
            self.data.append(data)

    def on_batch_begin(self, batch):
        # Set flag
        self._in_batch = True
        
        # Add a new list to each variable
        self.data.append([])
            
            
            
