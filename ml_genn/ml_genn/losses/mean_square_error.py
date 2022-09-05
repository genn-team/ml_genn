import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel


class MeanSquareError(Loss):
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add extra global parameter to store Y* throughout example
        egp_size = (example_timesteps * batch_size * np.prod(shape))
        model.add_egp("YTrue", "scalar*", np.empty(egp_size))
    
    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        # Check shape
        expected_shape = (example_timesteps, batch_size) + shape
        if y_true.shape != expected_shape:
            raise RuntimeError(f"Shape of target data for MeanSquareError "
                               f"loss should be {expected_shape}")
        
        # Copy flattened y_true into view
        genn_pop.extra_global_params["YTrue"].view[:] = y_true.flatten()
        
        # Push YTrue to device
        genn_pop.push_extra_global_param_to_device("YTrue")

