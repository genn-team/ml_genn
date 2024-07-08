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


logger = logging.getLogger(__name__)

class VarRecorder(Callback):
    """Callback used for recording state variables during simulation. 
    Variables can specified either by the name of the mlGeNN 
    :class:`ml_genn.utils.value.ValueDescriptor` class attribute corresponding to
    the variable e.g. ``v`` for the membrane voltage of a 
    :class:`ml_genn.neurons.LeakyIntegrateFire` neuron or by the internal name
    of a GeNN state variable e.g. ``LambdaV`` which is a state variable
    added to track gradients by :class:`ml_genn.compilers.EventPropCompiler`.
    
    Args:
        pop:            Population to record from
        var:            Name of variable to record
        key:            Key to assign recording data produced by this 
                        callback in dictionary  returned by 
                        evaluation/training methods of compiled network
        example_filter: Filter used to select which examples to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        neuron_filter:  Filter used to select which neurons to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        genn_var:       Internal name of variable to record
    """
    def __init__(self, pop: PopulationType, var: Optional[str] = None,
                 key=None, example_filter: ExampleFilterType = None,
                 neuron_filter: NeuronFilterType = None,
                 genn_var: Optional[str] = None):
        # Get underlying population
        self._pop = get_underlying_pop(pop)

        # Get the name of the GeNN variable corresponding to var
        if var is not None:
            self._var = get_genn_var_name(self._pop.neuron, var)
        elif genn_var is not None:
            self._var = genn_var
        else:
            raise RuntimeError("VarRecorder callback requires a "
                               "variable to be specified, either "
                               "via 'var' or 'genn_var' argument")

        # Stash key
        self.key = key

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # Create neuron filter mask
        self._neuron_mask = get_neuron_filter_mask(neuron_filter,
                                                   self._pop.shape)

    def set_params(self, data, compiled_network, **kwargs):
        self._batch_size = compiled_network.genn_model.batch_size
        self._compiled_network = compiled_network

        # Create default batch mask in case on_batch_begin not called
        self._batch_mask = np.ones(self._batch_size, dtype=bool)

        # Get GeNN population from compiled model
        pop = compiled_network.neuron_populations[self._pop]

        # Get neuron model variables
        pop_vars = pop.model.get_vars()

        try:
            # Find variable
            var = next(v for v in pop_vars if v.name == self._var)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self._var} to record")

        # Determine if var is shared or batched
        self.shared = not (get_var_access_dim(var.access)
                           & VarAccessDim.ELEMENT)
        self.batched = (get_var_access_dim(var.access)
                        & VarAccessDim.BATCH)
                           
        # If variable is shared and neuron mask was set, give warning
        if self.shared and not np.all(self._neuron_mask):
            logger.warn(f"VarRecorder ignoring neuron mask applied "
                        f"to SHARED_NEURON variable f{self._var}")

        # Create empty list to hold recorded data
        data[self.key] = []
        self._data = data[self.key]

    def on_timestep_end(self, timestep: int):
        # If anything should be recorded this batch
        if self._batch_count > 0:
            # Copy variable from device
            pop = self._compiled_network.neuron_populations[self._pop]
            pop.vars[self._var].pull_from_device()

            # If simulation and variable is batched
            var_view = pop.vars[self._var].current_view
            if self._batch_size > 1 and self.batched:
                # Apply neuron mask
                if self.shared:
                    data_view = var_view[self._batch_mask][:, :]
                else:
                    data_view = var_view[self._batch_mask][:, self._neuron_mask]
            # Otherwise
            else:
                # Apply neuron mask
                if self.shared:
                    data_view = var_view[:]
                else:
                    data_view = var_view[self._neuron_mask]

                # If SIMULATION is batched but variable isn't,
                # broadcast batch count copies of variable
                if self._batch_size > 1:
                    data_view = np.broadcast_to(
                        data_view, (self._batch_count,) + data_view.shape)

            # If there isn't already list to hold data, add one
            if len(self._data) == 0:
                self._data.append([])

            # Add copy to newest list
            self._data[-1].append(np.copy(data_view))

    def on_batch_begin(self, batch: int):
        # Get mask for examples in this batch and count
        self._batch_mask = self._example_filter.get_batch_mask(
            batch, self._batch_size)
        self._batch_count = np.sum(self._batch_mask)

        # If there's anything to record in this
        # batch, add list to hold it to data
        if self._batch_count > 0:
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
