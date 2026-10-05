from __future__ import annotations

import logging
import numpy as np

from pygenn import SynapseMatrixConnectivity, VarAccessDim
from .callback import Callback
from ..utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType

from dataclasses import dataclass, field
from itertools import chain
from pygenn import get_var_access_dim
from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_conn
from ..connection import Connection

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compilers import CompiledNetwork

logger = logging.getLogger(__name__)

@dataclass
class State:
    compiled_network: CompiledNetwork
    batched: bool
    batch_mask: int
    data: list = field(default_factory=list)
    batch_count: int = None

class ConnGradRecorder(Callback):
    """Callback used for recording connection state variables during
    simulation. Variables are specified using GeNN state variable name.
    By convention, ``g`` is always the weight and ``d`` is the per-synapse
    delay if it is used. Other variables are compiler-specific e.g. 
    ``Gradient`` accomulates gradients when using
    :class:`ml_genn.compilers.EventPropCompiler`.
    
    Args:
        conn:               Synapse population to record from
        genn_var:           Internal name of variable to record
        key:                Key to assign recording data produced by this
                            callback in dictionary returned by
                            evaluation/training methods of compiled network
        example_filter:     Filter used to select which examples to record
                            from (see :ref:`section-callbacks-recording`
                            for more information).        
    """
    def __init__(self, conn: Connection, genn_var: str,
                 key=None, example_filter: ExampleFilterType = None):
        # Get underlying connection
        self.conn = get_underlying_conn(conn)
        self.var = genn_var
        
        # Stash key
        self.key = key

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

    def create_state(self, compiled_network, **kwargs):
        # Get GeNN synapse group from compiled model
        pop = compiled_network.connection_populations[self.conn]

        # Get weight update model variables
        wum_vars = pop.wu_initialiser.snippet.get_vars()

        try:
            # Find variable
            var = next(v for v in wum_vars if v.name == self.var)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.var} to record")

        # Determine if var is batched
        batched = (get_var_access_dim(var.access) & VarAccessDim.BATCH)

        # Create state dataclass
        return State(compiled_network, batched,
                     np.ones(compiled_network.genn_model.batch_size,
                             dtype=bool))

    def on_timestep_end(self, state, timestep: int):
        # If anything should be recorded this batch
        cn = state.compiled_network
        if state.batch_count > 0 and timestep == state.compiled_network.example_timesteps-1:
            # Copy variable from device
            pop = cn.connection_populations[self.conn]
            pop.vars[self.var].pull_from_device()

            # Get view, sliced by batch mask if simulation is batched
            var_view = pop.vars[self.var].view

            # If simulation and variable is batched
            if cn.genn_model.batch_size > 1 and state.batched:
                # Slice view by batch mask 
                data_view = np.reshape(var_view[state.batch_mask],
                                       (state.batch_count, -1))                
            else:
                data_view = var_view
                
                # If SIMULATION is batched but variable isn't,
                # broadcast batch count copies of variable
                if cn.genn_model.batch_size > 1:
                    data_view = np.broadcast_to(
                        data_view, (state.batch_count,) + data_view.shape)

            # Add copy to newest list
            state.data.append(np.copy(data_view))

    def on_batch_begin(self, state, batch: int):
        # Get mask for examples in this batch and count
        state.batch_mask = self._example_filter.get_batch_mask(
            batch, state.compiled_network.genn_model.batch_size)
        state.batch_count = np.sum(state.batch_mask)

    def get_data(self, state):
        # Stack 1D or 2D numpy arrays containing value
        # for each timestep in each batch to get
        # (time, batch, neuron) or (time, neuron) arrays
        data = np.asarray(state.data) 
        data= np.reshape(data,(-1,data.shape[-1]))
                         
        return self.key, data
