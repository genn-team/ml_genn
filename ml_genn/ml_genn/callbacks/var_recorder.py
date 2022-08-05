import numpy as np

from itertools import chain
from typing import Sequence, Union

from ..utils.network import get_underlying_pop

class VarRecorder:
    def __init__(self, pop, var: str):
        # Get underlying population
        # **TODO** handle Connection variables as well
        self._pop = get_underlying_pop(pop)
        self._var = var
        
        # Create empty list to hold recorded data
        self._data = [[]]
    
    def set_params(self, compiled_network, **kwargs):
        self._compiled_network = compiled_network
            
    def on_timestep_end(self, timestep):
        # Copy variable from device
        pop = self._compiled_network.neuron_populations[self._pop]
        pop.pull_var_from_device(self._var)
        
        # Add data to newest list
        self._data[-1].append(np.copy(pop.vars[self._var].view))
        
    def on_batch_begin(self, batch):
        # If this isn't the first batch where the constructor 
        # will have already added a list, add a new list to data
        if batch > 0:
            self._data.append([])
        
    @property
    def data(self):
        # If simulation is example based
        batched = (self._compiled_network.genn_model.batch_size > 1)

        # Stack 1D or 2D numpy arrays containing value 
        # for each timestep in each batch to get 
        # (time, batch, neuron) or (time, neuron) arrays
        data = [np.stack(d) for d in self._data]
        
        # If model batched
        if batched:
            # Split each stacked array along the batch axis and 
            # chain together resulting in a list, containing a 
            # (time, neuron) matrix for each example
            data = list(chain.from_iterable(np.split(d, d.shape[1], axis=1)
                                            for d in data))
            
            # Finally, remove the batch axis from each matrix
            # (which will now always be size 1) and return
            data = [np.squeeze(d, axis=1) for d in data]
            return data
        # Otherwise, return data directly
        else:
            return data
