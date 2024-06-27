import logging
import numpy as np

from itertools import chain

from pygenn import VarAccessDim
from typing import Optional
from .callback import Callback
from ..utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType
from ..utils.network import PopulationType

from pygenn import get_var_access_dim
from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_pop
from ..utils.value import get_genn_var_name
from ..connection import Connection

logger = logging.getLogger(__name__)

class ConnVarRecorder(Callback):
    """*TODO* UPDATE Callback used for recording connection state variables during simulation. 
    Variables can specified either by the name of the mlGeNN 
    :class:`ml_genn.utils.value.ValueDescriptor` class attribute corresponding to
    the variable e.g. ``v`` for the membrane voltage of a 
    :class:`ml_genn.neurons.LeakyIntegrateFire` neuron or by the internal name
    of a GeNN state variable e.g. ``LambdaV`` which is a state variable
    added to track gradients by :class:`ml_genn.compilers.EventPropCompiler`.
    
    Args:
        pop:            Synapse population to record from
        var:            Name of variable to record
        key:            Key to assign recording data produced by this 
                        callback in dictionary  returned by 
                        evaluation/training methods of compiled network
        example_filter: Filter used to select which examples to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        *TODO* synapse_filter:  Filter used to select which synapses to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        genn_var:       Internal name of variable to record
    """
    def __init__(self, pop: Connection, var: Optional[str] = None,
                 key=None, example_filter: ExampleFilterType = None,
                 #*TODO* synapse_filter: NeuronFilterType = None,
                 genn_var: Optional[str] = None):
        # Get underlying population
        # **TODO** handle Connection variables as well
        self._pop = pop

        # Get the name of the GeNN variable corresponding to var
        if var is not None:
            self._var = get_genn_var_name(self._pop.synapse, var)
        elif genn_var is not None:
            self._var = genn_var
        else:
            raise RuntimeError("ConnVarRecorder callback requires a "
                               "variable to be specified, either "
                               "via 'var' or 'genn_var' argument")

        # Stash key
        self.key = key

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # *TODO* Create neuron filter mask
        #self._neuron_mask = get_neuron_filter_mask(neuron_filter,
        #                                           self._pop.shape)

    def set_params(self, data, compiled_network, **kwargs):
        self._batch_size = compiled_network.genn_model.batch_size
        self._compiled_network = compiled_network

        # Create default batch mask in case on_batch_begin not called
        self._batch_mask = np.ones(self._batch_size, dtype=bool)

        try:
            # Get GeNN population from compiled model
            pop = compiled_network.connection_populations[self._pop]

            # Get neuronmodel variables
            print(dir(pop))
            pop_vars = pop.vars

            # Find variable
            for v in pop_vars:
                print(v)
            var = next(v for v in pop_vars if v == self._var)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self._var} to record")

        # *TODO* Determine if var is shared
        #self.shared = not (get_var_access_dim(var.access) & VarAccessDim.ELEMENT)

        # If variable is shared and neuron mask was set, give warning
        #if self.shared and not np.all(self._neuron_mask):
        #    logger.warn(f"VarRecorder ignoring neuron mask applied "
        #                f"to SHARED_NEURON variable f{self._var}")

        # Create empty list to hold recorded data
        data[self.key] = []
        self._data = data[self.key]

    def on_timestep_end(self, timestep: int):
        # If anything should be recorded this batch
        if np.any(self._batch_mask):
            # Copy variable from device
            pop = self._compiled_network.connection_populations[self._pop]
            pop.vars[self._var].pull_from_device()

            # Get view, sliced by batch mask if simulation is batched
            var_view = pop.vars[self._var].view
            #*TODO* consider shared vars
            if self._batch_size > 1:
                data_view = var_view[self._batch_mask][:, :]
            else:
                data_view = var_view[:]

            # If there isn't already list to hold data, add one
            if len(self._data) == 0:
                self._data.append([])

            # Add copy to newest list
            self._data[-1].append(np.copy(data_view))

    def on_batch_begin(self, batch: int):
        # Get mask for examples in this batch
        self._batch_mask = self._example_filter.get_batch_mask(
            batch, self._batch_size)

        # If there's anything to record in this
        # batch, add list to hold it to data
        if np.any(self._batch_mask):
            self._data.append([])

    def get_data(self):
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

        return self.key, data
