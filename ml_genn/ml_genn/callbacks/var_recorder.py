from __future__ import annotations

import logging
import numpy as np

from itertools import chain

from pygenn import VarAccess, VarAccessDim
from typing import Union
from .callback import Callback
from .. import InputLayer, Layer, Population
from ..utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType
from ..utils.network import ConnectionType, PopulationType

from dataclasses import dataclass, field
from pygenn import get_var_access_dim
from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_conn, get_underlying_pop

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compilers import CompiledNetwork

logger = logging.getLogger(__name__)

def _find_var(model_vars, name):
    try:
        # Find variable
        return next(v for v in model_vars if v.name == name)
    except StopIteration:
        raise RuntimeError(f"Model does not have variable "
                           f"{name} to record")
@dataclass
class State:
    compiled_network: CompiledNetwork
    batched: bool
    shared: bool
    batch_mask: int
    data: list = field(default_factory=list)
    batch_count: int = None
    

class VarRecorder(Callback):
    """Callback used for recording state variables during simulation. 

    Args:
        pop:            Population to record neuron variables or
                        Connection to record synapse variables from
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

    """
    def __init__(self, pop: Union[PopulationType, ConnectionType], var: str,
                 key=None, example_filter: ExampleFilterType = None,
                 neuron_filter: NeuronFilterType = None):
        # Get underlying population or connection
        # **YUCK** in Python 3.10 could just isinstance(PopulationType)
        if isinstance(pop, (InputLayer, Layer, Population)):
            self._pop = get_underlying_pop(pop)
            var_shape = self._pop.shape
        else:
            self._pop = get_underlying_conn(pop)
            var_shape = self._pop.target().shape

        # Stash var name and key
        self.var = var
        self.key = key

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # Create neuron filter mask
        self._neuron_mask = get_neuron_filter_mask(neuron_filter,
                                                   var_shape)

    def create_state(self, compiled_network, **kwargs):
        if isinstance(self._pop, Population):
            # Get GeNN neuron population from compiled model
            pop = compiled_network.neuron_populations[self._pop]
            
            # Check variable exists and get its access mode
            var_access = _find_var(pop.model.get_vars(), self.var).access
        else:
            # Get GeNN synapse population from compiled model
            pop = compiled_network.connection_populations[self._pop]

            if self.var == "out_post":
                var_access = VarAccess.READ_WRITE
            else:
                # Check variable exists and get its access mode
                var_access = _find_var(
                    pop.ps_initialiser.snippet.get_vars(), self.var).access
            
   
        # Determine if var is shared or batched
        shared = not (get_var_access_dim(var_access) & VarAccessDim.ELEMENT)
        batched = (get_var_access_dim(var_access) & VarAccessDim.BATCH)
                           
        # If variable is shared and neuron mask was set, give warning
        if shared and not np.all(self._neuron_mask):
            logger.warning(f"VarRecorder ignoring neuron mask applied "
                           f"to SHARED_NEURON variable {self.var}")

        return State(compiled_network, batched, shared,
                     np.ones(compiled_network.genn_model.batch_size,
                             dtype=bool))

    def on_timestep_end(self, state, timestep: int):
        # If anything should be recorded this batch
        cn = state.compiled_network
        if state.batch_count > 0:
            if isinstance(self._pop, Population):
                pop = cn.neuron_populations[self._pop]
                var = pop.vars[self.var]
                var_view = var.current_view
            else:
                pop = cn.connection_populations[self._pop]
                if self.var == "out_post":
                    var = pop.out_post
                    var_view = var.view
                else:
                    var = pop.psm_vars[self.var]
                    var_view = var.current_view
            
            # Copy variable from device
            var.pull_from_device()

            # If simulation and variable is batched
            if cn.genn_model.batch_size > 1 and state.batched:
                # Apply neuron mask
                if state.shared:
                    data_view = var_view[state.batch_mask][:, :]
                else:
                    data_view = var_view[state.batch_mask][:, self._neuron_mask]
            # Otherwise
            else:
                # Apply neuron mask
                if state.shared:
                    data_view = var_view[:]
                else:
                    data_view = var_view[self._neuron_mask]

                # If SIMULATION is batched but variable isn't,
                # broadcast batch count copies of variable
                if cn.genn_model.batch_size > 1:
                    data_view = np.broadcast_to(
                        data_view, (state.batch_count,) + data_view.shape)

            # If there isn't already list to hold data, add one
            if len(state.data) == 0:
                state.data.append([])

            # Add copy to newest list
            state.data[-1].append(np.copy(data_view))

    def on_batch_begin(self, state, batch: int):
        # Get mask for examples in this batch and count
        state.batch_mask = self._example_filter.get_batch_mask(
            batch, state.compiled_network.genn_model.batch_size)
        state.batch_count = np.sum(state.batch_mask)

        # If there's anything to record in this
        # batch, add list to hold it to data
        if state.batch_count > 0:
            state.data.append([])

    def get_data(self, state):
        # Stack 1D or 2D numpy arrays containing value
        # for each timestep in each batch to get
        # (time, batch, neuron) or (time, neuron) arrays
        data = [np.stack(d) for d in state.data]

        # If model batched
        if state.compiled_network.genn_model.batch_size > 1:
            # Split each stacked array along the batch axis and
            # chain together resulting in a list, containing a
            # (time, neuron) matrix for each example
            data = list(chain.from_iterable(np.split(d, d.shape[1], axis=1)
                                            for d in data))

            # Finally, remove the batch axis from each matrix
            # (which will now always be size 1) and return
            data = [np.squeeze(d, axis=1) for d in data]

        return self.key, data