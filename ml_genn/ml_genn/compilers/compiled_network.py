from typing import Sequence, Union
from ..population import Population
from ..layer import InputLayer, Layer
from ..utils.callback_list import CallbackList

from ..utils.network import get_underlying_pop


class CompiledNetwork:
    _context = None

    def __init__(self, genn_model, neuron_populations,
                 connection_populations, num_recording_timesteps=None):

        self.genn_model = genn_model
        self.neuron_populations = neuron_populations
        self.connection_populations = connection_populations
        self.num_recording_timesteps = num_recording_timesteps

    def set_input(self, inputs: dict):
        # Loop through populations
        for pop, input in inputs.items():
            # Find corresponding GeNN population and set input
            pop = get_underlying_pop(pop)
            pop.neuron.set_input(self.neuron_populations[pop],
                                 self.genn_model.batch_size, pop.shape, input)

    def get_output(self, outputs: Union[Sequence, Population,
                                        InputLayer, Layer]):
        if isinstance(outputs, Sequence):
            return [self._get_output(p) for p in outputs]
        else:
            return self._get_output(outputs)

    def custom_update(self, name: str):
        """Perform custom update"""
        self.genn_model.custom_update(name)

    def step_time(self, callback_list: CallbackList=None):
        """Step the GeNN model
        """
        if callback_list is not None:
            callback_list.on_timestep_begin()
        
        self.genn_model.step_time()
        
        if callback_list is not None:
            callback_list.on_timestep_end()

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

    def _get_output(self, pop):
        pop = get_underlying_pop(pop)
        return pop.neuron.get_output(self.neuron_populations[pop],
                                     self.genn_model.batch_size, pop.shape)
