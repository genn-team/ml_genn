import numpy as np

from itertools import chain
from typing import Sequence, Union

from .callback import Callback
from ..utils.filter import ExampleFilter

from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_pop

class VarRecorder(Callback):
    def __init__(self, pop, var: str, example_filter=None, neuron_filter=None):
        # Get underlying population
        # **TODO** handle Connection variables as well
        self._pop = get_underlying_pop(pop)
        self._var = var
        
        # Create example filter
        self._example_filter = ExampleFilter(example_filter)
        
        # Create neuron filter mask
        self._neuron_mask = get_neuron_filter_mask(neuron_filter, self._pop.shape)

        # Create empty list to hold recorded data
        self._data = []
    
    def set_params(self, compiled_network, **kwargs):
        self._batch_size = compiled_network.genn_model.batch_size
        self._compiled_network = compiled_network
        
        # Create default batch mask in case on_batch_begin not called
        self._batch_mask = np.ones(self._batch_size, dtype=bool)
            
    def on_timestep_end(self, timestep):
        # If anything should be recorded this batch
        if np.any(self._batch_mask):
            # Copy variable from device
            pop = self._compiled_network.neuron_populations[self._pop]
            pop.pull_var_from_device(self._var)
            
            # Get view, sliced by batch mask if simulation is batched
            var_view = pop.vars[self._var].view
            if self._batch_size > 1:
                data_view = var_view[self._batch_mask][:,self._neuron_mask]
            else:
                data_view = var_view[self._neuron_mask]
            
            # If there isn't already list to hold data, add one
            if len(self._data) == 0:
                self._data.append([])
            
            # Add copy to newest list
            self._data[-1].append(np.copy(data_view))
        
    def on_batch_begin(self, batch):  
        # Get mask for examples in this batch
        self._batch_mask = self._example_filter.get_batch_mask(
            batch, self._batch_size)
        
        # If there's anything to record in this 
        # batch, add list to hold it to data
        if np.any(self._batch_mask):
            self._data.append([])
        
    @property
    def data(self):
        # Stack 1D or 2D numpy arrays containing value 
        # for each timestep in each batch to get 
        # (time, batch, neuron) or (time, neuron) arrays
        data = [np.stack(d) for d in self._data]
        
        # If model batched
        if self._batch_size > 1:
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
