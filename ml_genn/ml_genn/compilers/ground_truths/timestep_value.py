import numpy as np

from .ground_truth import GroundTruth
from ml_genn.utils.model import NeuronModel


class TimestepValue(GroundTruth):
    """Ground truth in the form of a tensor of values, output neuron
    readouts aim to produce each timestep"""
    def add_to_neuron(self, backward: bool, model: NeuronModel, 
                      shape, batch_size: int, example_timesteps: int):
        # Add extra global parameter to store Y* throughout example
        flat_shape = np.prod(shape)
        egp_size = (example_timesteps * batch_size * flat_shape)
        model.add_egp("YTrue", "scalar*",
                      np.empty(egp_size, dtype=np.float32))

    def push_to_device(self, genn_pop, y_true, shape, batch_size: int, 
                       example_timesteps: int):
        # Check shape
        expected_shape = (example_timesteps,) + shape
        y_true = np.asarray(y_true)
        if y_true.shape[1:] != expected_shape or len(y_true) > batch_size:
            raise RuntimeError(f"Shape of target data for TimestepValue "
                               f"ground truth should be {expected_shape}")

        # Copy flattened y_true into (1D) view
        y_true_flat = y_true.flatten()
        egp = genn_pop.extra_global_params["YTrue"]
        egp.view[:len(y_true_flat)] = y_true_flat

        # Push YTrue to device
        egp.push_to_device()
