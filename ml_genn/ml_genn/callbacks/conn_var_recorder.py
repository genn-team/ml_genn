import logging
import numpy as np

from itertools import chain

from pygenn import SynapseMatrixConnectivity, VarAccessDim
from typing import Optional
from .callback import Callback
from ..utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType
from ..utils.network import ConnectionType

from pygenn import get_var_access_dim
from ..utils.filter import get_neuron_filter_mask
from ..utils.network import get_underlying_conn
from ..connection import Connection

logger = logging.getLogger(__name__)

class ConnVarRecorder(Callback):
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
        src_neuron_filter:  Filter used to select which synapses to record
                            from (see :ref:`section-callbacks-recording` 
                            for more information).
        trg_neuron_filter:  Filter used to select which synapses to record
                            from (see :ref:`section-callbacks-recording` 
                            for more information).
        
    """
    def __init__(self, conn: Connection, genn_var: str,
                 key=None, example_filter: ExampleFilterType = None,
                 src_neuron_filter: NeuronFilterType = None,
                 trg_neuron_filter: NeuronFilterType = None):
        # Get underlying connection
        self._conn = get_underlying_conn(conn)
        self._var = genn_var
        
        # Stash key
        self.key = key

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # Create neuron filter masks for source and target populations
        self._src_neuron_mask = get_neuron_filter_mask(
            src_neuron_filter, self._conn.source().shape)
        self._trg_neuron_mask = get_neuron_filter_mask(
            trg_neuron_filter, self._conn.target().shape)

    def set_params(self, data, compiled_network, **kwargs):
        self._batch_size = compiled_network.genn_model.batch_size
        self._compiled_network = compiled_network

        # Create default batch mask in case on_batch_begin not called
        self._batch_mask = np.ones(self._batch_size, dtype=bool)

        # Get GeNN synapse group from compiled model
        pop = compiled_network.connection_populations[self._conn]

        # Give error if connectivity is not dense
        if not pop.matrix_type & SynapseMatrixConnectivity.DENSE:
            raise NotImplementedError("ConnVarRecorder currently only "
                                      "supports recording variables "
                                      "associated with DENSE connectivity")

        # Get weight update model variables
        wum_vars = pop.wu_initialiser.snippet.get_vars()

        try:
            # Find variable
            var = next(v for v in wum_vars if v.name == self._var)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self._var} to record")

        # Determine if var is batched
        self.batched = (get_var_access_dim(var.access)
                        & VarAccessDim.BATCH)
                        
        # Create empty list to hold recorded data
        data[self.key] = []
        self._data = data[self.key]

    def on_timestep_end(self, timestep: int):
        # If anything should be recorded this batch
       if self._batch_count > 0:
            # Copy variable from device
            pop = self._compiled_network.connection_populations[self._conn]
            pop.vars[self._var].pull_from_device()

            # Get view, sliced by batch mask if simulation is batched
            var_view = pop.vars[self._var].view

            # If simulation and variable is batched
            num_src = np.prod(self._conn.source().shape)
            num_trg = np.prod(self._conn.target().shape)
            if self._batch_size > 1 and self.batched:
                # Slice view by batch mask and reshape to num_src * num_trg
                data_view = np.reshape(var_view[self._batch_mask],
                                       (self._batch_count, num_src, num_trg))
                
                # Slice view with src and target masks
                data_view = data_view[:,self._src_neuron_mask,:]
                data_view = data_view[:,:,self._trg_neuron_mask]
            else:
                # Reshape view to num_src * num_trg
                data_view = np.reshape(var_view, (num_src, num_trg))

                # Slice view with src and target masks
                data_view = data_view[self._src_neuron_mask]
                data_view = data_view[:,self._trg_neuron_mask]

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
