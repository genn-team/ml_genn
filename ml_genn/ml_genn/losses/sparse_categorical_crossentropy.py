import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel


class SparseCategoricalCrossentropy(Loss):
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add extra global parameter to store labels for each neuron
        model.add_egp("YTrue", "uint8_t*", 
                      np.empty(batch_size))

        # Add sim-code to convert label to one-hot
        model.append_sim_code(
            f"""
            const scalar yTrue = ($(id) == $(YTrue)[$(batch)]) ? 1.0 : 0.0;
            """)

    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        # Check shape
        y_true = np.asarray(y_true)
        if y_true.ndim != 1 or len(y_true) > batch_size:
            raise RuntimeError(f"Length of target data for "
                               f"SparseCategoricalCrossentropy loss should "
                               f"be < {batch_size}")
        
        # Copy flattened y_true into view
        genn_pop.extra_global_params["YTrue"].view[:len(y_true)] = y_true
        
        # Push YTrue to device
        genn_pop.push_extra_global_param_to_device("YTrue")
