import numpy as np

from typing import List, Mapping, Optional, Sequence, Union
from ..utils.callback_list import CallbackList
from ..utils.network import PopulationType

from ..utils.network import get_underlying_pop

OutputType = Union[np.ndarray, List[np.ndarray]]


class CompiledNetwork:
    _context = None

    def __init__(self, genn_model, neuron_populations,
                 connection_populations, communicator,
                 num_recording_timesteps=None):
        self.genn_model = genn_model
        self.neuron_populations = neuron_populations
        self.connection_populations = connection_populations
        self.communicator = communicator
        self.num_recording_timesteps = num_recording_timesteps

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

        # Build model (only on first rank if there is a communicator)
        first_rank = (self.communicator is None 
                      or self.communicator.rank == 0)
        if first_rank and not self.genn_model._built:
            self.genn_model.build()

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
