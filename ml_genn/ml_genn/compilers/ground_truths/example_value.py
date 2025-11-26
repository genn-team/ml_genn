import numpy as np

from .ground_truth import GroundTruth
from ml_genn.utils.model import NeuronModel
from pygenn import VarAccess
from typing import List, Tuple

class ExampleValue(GroundTruth):
    """Ground truth in the form of a value, output neuron
    readouts aim to produce at the end of each example"""
    def add_to_neuron(self, backward: bool, model: NeuronModel, 
                      shape, batch_size: int, example_timesteps: int):
        # Add variable, shared across neurons to hold true value for batch
        model.add_var("YTrue", "scalar", 0.0, 
                      VarAccess.READ_ONLY_DUPLICATE, reset=False)
        
        # If backward pass is required, add second variable to 
        # hold the true value for the backward pass
        if backward:
            model.add_var("YTrueBack", "scalar", 0.0,
                          VarAccess.READ_ONLY_DUPLICATE, reset=False)

    def push_to_device(self, genn_pop, y_true, shape, batch_size: int,
                       example_timesteps: int):
        # Check shape
        y_true = np.asarray(y_true)
        if y_true.shape[1:] != shape or len(y_true) > batch_size:
            raise RuntimeError(f"Shape of target data for ExampleValue "
                               f"ground truth should be {expected_shape}")

        # Copy flattened y_true into (2D) view
        if batch_size == 1:
            genn_pop.vars["YTrue"].view[:] = y_true.flatten()
        else:
            genn_pop.vars["YTrue"].view[:len(y_true), :] = y_true

        # Push YTrue to device
        genn_pop.vars["YTrue"].push_to_device()

    @property
    def backward_duplicate_var_reset(self) -> List[Tuple[str, str, str]]:
        """
        Gets resets for any per-neuron variables this ground truth adds
        """
        return [("YTrueBack", "scalar", "YTrue")]