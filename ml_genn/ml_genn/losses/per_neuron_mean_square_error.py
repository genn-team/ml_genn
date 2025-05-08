import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel


class PerNeuronMeanSquareError(Loss):
    """Computes the mean squared error between labels and prediction"""
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add variable, shared across neurons to hold true label for batch
        model.add_var("YTrue", "scalar", 0.0)

        # Add sim-code to convert label to one-hot
        # **THINK** pointless
        model.append_sim_code(
            f"""
            const scalar yTrue = YTrue;
            """)

    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        # Check shape
        expected_shape = (batch_size,) + shape
        y_true = np.asarray(y_true)
        if y_true.shape != expected_shape:
            raise RuntimeError(f"Shape of target data for PerNeuronMeanSquareError "
                               f"loss should be {expected_shape}")

        # Copy flattened y_true into view
        genn_pop.vars["YTrue"].view[:batch_size, :] = y_true

        # Push YTrue to device
        genn_pop.vars["YTrue"].push_to_device()

