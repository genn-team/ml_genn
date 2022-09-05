import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel


class SparseCategoricalCrossentropy(Loss):
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add extra global parameter to store labels for each neuron
        model.add_egp("YTrue", "uint8_t*", 
                      np.empty(batch_size))
    
    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        # Check shape
        expected_shape = (batch_size,)
        if y_true.shape != expected_shape:
            raise RuntimeError(f"Shape of target data for "
                               f"SparseCategoricalCrossentropy loss should "
                               f"be {expected_shape}")
        
        # Copy flattened y_true into view
        genn_pop.extra_global_params["YTrue"].view[:] = y_true.flatten()
        
        # Push YTrue to device
        genn_pop.push_extra_global_param_to_device("YTrue")
