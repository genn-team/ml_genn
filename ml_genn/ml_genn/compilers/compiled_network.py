import numpy as np

from typing import List, Mapping, Optional, Sequence, Union
from ..serialisers import Serialiser
from ..utils.callback_list import CallbackList
from ..utils.network import PopulationType

from ..utils.module import get_object
from ..utils.network import get_underlying_pop

from ..serialisers import default_serialisers

OutputType = Union[np.ndarray, List[np.ndarray]]


class CompiledNetwork:
    _context = None

    def __init__(self, genn_model, neuron_populations,
                 connection_populations, communicator,
                 num_recording_timesteps=None,
                 checkpoint_connection_vars: list = [],
                 checkpoint_population_vars: list = []):
        self.genn_model = genn_model
        self.neuron_populations = neuron_populations
        self.connection_populations = connection_populations
        self.communicator = communicator
        self.num_recording_timesteps = num_recording_timesteps
        self.checkpoint_connection_vars = checkpoint_connection_vars
        self.checkpoint_population_vars = checkpoint_population_vars

        # Build set of synapse groups with checkpoint variables
        self.checkpoint_synapse_groups = set(
            connection_populations[c] 
            for c, _ in self.checkpoint_connection_vars)
    
    def save_connectivity(self, keys=(), serialiser="numpy"):
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)
        
        # Loop through connections and their corresponding synapse groups
        for c, genn_pop in self.connection_populations.items():
            # If synapse group has ragged connectivity, download  
            # connectivity and save pre and postsynaptic indices
            if genn_pop.is_ragged:
                genn_pop.pull_connectivity_from_device()
                serialiser.serialise(keys + (c, "pre_ind"),
                                     genn_pop.get_sparse_pre_inds())
                serialiser.serialise(keys + (c, "post_ind"),
                                     genn_pop.get_sparse_post_inds())

    def save(self, keys=(), serialiser="numpy"):
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)
        
        # Loop through synapse groups with variables to be checkpointed
        for genn_pop in self.checkpoint_synapse_groups:
            # If synapse group has ragged connectivity, download  
            # connectivity so variables can be accessed correctly
            if genn_pop.is_ragged:
                genn_pop.pull_connectivity_from_device()

        # Loop through connection variables to checkpoint
        for c, v in self.checkpoint_connection_vars:
            genn_pop = self.connection_populations[c]
            genn_pop.pull_var_from_device(v)
            serialiser.serialise(keys + (c, v), genn_pop.get_var_values(v))

        # Loop through population variables to checkpoint
        for p, v in self.checkpoint_population_vars:
            genn_pop = self.neuron_populations[p]
            genn_pop.pull_var_from_device(v)
            serialiser.serialise(keys + (p, v), genn_pop.vars[v].view)

    def set_input(self, inputs: dict):
        # Loop through populations
        for pop, input in inputs.items():
            # Find corresponding GeNN population and set input
            pop = get_underlying_pop(pop)
            pop.neuron.set_input(self.neuron_populations[pop],
                                 self.genn_model.batch_size, pop.shape, input)

    def get_readout(self, outputs: Union[Sequence, PopulationType]) -> OutputType:
        if isinstance(outputs, Sequence):
            return [self._get_readout(p) for p in outputs]
        else:
            return self._get_readout(outputs)

    def custom_update(self, name: str):
        """Perform custom update"""
        self.genn_model.custom_update(name)

    def step_time(self, callback_list: Optional[CallbackList] = None):
        """Step the GeNN model
        """
        if callback_list is not None:
            callback_list.on_timestep_begin(self.genn_model.timestep)

        self.genn_model.step_time()

        if callback_list is not None:
            callback_list.on_timestep_end(self.genn_model.timestep - 1)

    def reset_time(self):
        """Reset the GeNN model"""
        self.genn_model.timestep = 0
        self.genn_model.t = 0.0

    def __enter__(self):
        if CompiledNetwork._context is not None:
            raise RuntimeError("Nested compiled networks are "
                               "not currently supported")
        CompiledNetwork._context = self

        # If model, isn't already built
        first_rank = (self.communicator is None 
                      or self.communicator.rank == 0)
        if not self.genn_model._built:
            # If this is the first rank, build model
            if first_rank:
                self.genn_model.build()
            # Otherwise, at least ensure it is finalised
            # **HACK** GeNN should handle this
            else:
                self.genn_model._model.finalize()

        # If there is a communicator, wait for all ranks to reach this point
        if self.communicator is not None:
            self.communicator.barrier()

        self.genn_model.load(
            num_recording_timesteps=self.num_recording_timesteps)

        # If there is a communicator
        if self.communicator is not None:
            # Generate unique ID for our NCCL 'clique' on first rank
            if first_rank:
                self.genn_model._slm.nccl_generate_unique_id()

            # Broadcast our  NCCL clique ID across all ranks
            nccl_unique_id_view =\
                self.genn_model._slm.nccl_assign_external_unique_id()
            self.communicator.broadcast(nccl_unique_id_view, 0)

            # Initialise NCCL communicator
            self.genn_model._slm.nccl_init_communicator(
                self.communicator.rank, self.communicator.num_ranks)

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert CompiledNetwork._context is not None
        CompiledNetwork._context = None

        self.genn_model.unload()

    def _get_readout(self, pop: PopulationType) -> np.ndarray:
        pop = get_underlying_pop(pop)
        return pop.neuron.get_readout(self.neuron_populations[pop],
                                      self.genn_model.batch_size, pop.shape)
