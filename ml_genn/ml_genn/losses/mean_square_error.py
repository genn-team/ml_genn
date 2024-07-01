import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel


class MeanSquareError(Loss):
    """Computes the mean squared error between labels and prediction"""
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add extra global parameter to store Y* throughout example
        flat_shape = np.prod(shape)
        egp_size = (example_timesteps * batch_size * flat_shape)
        model.add_egp("YTrue", "scalar*",
                      np.empty(egp_size, dtype=np.float32))

        # Add sim-code to read out correct yTrue value 
        model.append_sim_code(
            f"""
            const unsigned int timestep = (int)round(t / dt);
            const unsigned int index = (batch * {example_timesteps} * num_neurons)
                                       + (timestep * num_neurons) + id;
            const scalar yTrue = YTrue[index];
            """)

    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        # Check shape
        expected_shape = (batch_size, example_timesteps) + shape
        y_true = np.asarray(y_true)
        if y_true.shape != expected_shape:
            raise RuntimeError(f"Shape of target data for MeanSquareError "
                               f"loss should be {expected_shape}")

        # Copy flattened y_true into view
        genn_pop.extra_global_params["YTrue"].view[:] = y_true.flatten()

        # Push YTrue to device
        genn_pop.extra_global_params["YTrue"].push_to_device()

