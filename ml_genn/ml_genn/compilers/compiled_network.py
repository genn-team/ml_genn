import numpy as np

from typing import List, Mapping, Optional, Sequence, Union
from ..utils.callback_list import CallbackList
from ..utils.network import PopulationType

from ..utils.network import get_underlying_pop

OutputType = Union[np.ndarray, List[np.ndarray]]


class CompiledNetwork:
    """Base class for all compiled networks."""
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
        """Copy input data to GPU
        
        Args:
            inputs: Dictionary mapping input populations or 
                    layers to data to copy to them
        """
        # Loop through populations
        for pop, input in inputs.items():
            # Find corresponding GeNN population and set input
            pop = get_underlying_pop(pop)
            pop.neuron.set_input(self.neuron_populations[pop],
                                 self.genn_model.batch_size, pop.shape, input)

    def get_readout(self, outputs: Union[Sequence, PopulationType]) -> OutputType:
        """Get output from population readouts"""
        if isinstance(outputs, Sequence):
            return [self._get_readout(p) for p in outputs]
        else:
            return self._get_readout(outputs)

    def custom_update(self, name: str):
        """Perform custom update.
        
        Args:
            name:   Name of custom update"""
        self.genn_model.custom_update(name)

    def step_time(self, callback_list: Optional[CallbackList] = None):
        """Simulate one timestep
        
        Args:
            callback_list:  Callbacks to potentially execute 
                            at start and end of timestep
        """
        if callback_list is not None:
            callback_list.on_timestep_begin(self.genn_model.timestep)

        self.genn_model.step_time()

        if callback_list is not None:
            callback_list.on_timestep_end(self.genn_model.timestep - 1)

    def reset_time(self):
        """Reset the GeNN models internal timestep to 0."""
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
            # Otherwise, build but ensure no code generated
            else:
                self.genn_model.build(never_rebuild=True)

        # If there is a communicator, wait for all ranks to reach this point
        if self.communicator is not None:
            self.communicator.barrier()

        self.genn_model.load(
            num_recording_timesteps=self.num_recording_timesteps)

        # If there is a communicator
        if self.communicator is not None:
            # Get CUDA-specific state from runtime
            runtime_state = self.genn_model._runtime.state

            # Generate unique ID for our NCCL 'clique' on first rank
            if first_rank:
                runtime_state.nccl_generate_unique_id()

            # Broadcast our  NCCL clique ID across all ranks
            self.communicator.broadcast(runtime_state.nccl_unique_id, 0)

            # Initialise NCCL communicator
            runtime_state.nccl_init_communicator(
                self.communicator.rank, self.communicator.num_ranks)

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert CompiledNetwork._context is not None
        CompiledNetwork._context = None

        self.genn_model.unload()

    def _get_readout(self, pop: PopulationType) -> np.ndarray:
        pop = get_underlying_pop(pop)
        return pop.neuron.get_readout(self.neuron_populations[pop],
                                      self.genn_model.batch_size, pop.shape)
