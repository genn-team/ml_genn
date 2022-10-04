import numpy as np

from typing import List, Mapping, Optional, Sequence, Union
from ..utils.callback_list import CallbackList
from ..utils.network import PopulationType

from ..utils.network import get_underlying_pop

OutputType = Union[np.ndarray, List[np.ndarray]]


class CompiledNetwork:
    _context = None

    def __init__(self, genn_model, neuron_populations,
                 connection_populations, softmax,
                 num_recording_timesteps=None):
        self.genn_model = genn_model
        self.neuron_populations = neuron_populations
        self.connection_populations = connection_populations
        self.softmax = softmax
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

        # Calculate softmax if required
        if self.softmax:
            self.genn_model.custom_update("Softmax1")
            self.genn_model.custom_update("Softmax2")
            self.genn_model.custom_update("Softmax3")

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
        self.genn_model.build()
        self.genn_model.load(
            num_recording_timesteps=self.num_recording_timesteps)

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert CompiledNetwork._context is not None
        CompiledNetwork._context = None

    def _get_readout(self, pop: PopulationType) -> np.ndarray:
        pop = get_underlying_pop(pop)
        return pop.neuron.get_readout(self.neuron_populations[pop],
                                      self.genn_model.batch_size, pop.shape)
