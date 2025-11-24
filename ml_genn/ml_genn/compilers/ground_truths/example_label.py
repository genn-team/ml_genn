import numpy as np

from .ground_truth import GroundTruth
from ml_genn.utils.model import NeuronModel
from pygenn import VarAccess
from typing import List, Tuple


class ExampleLabel(GroundTruth):
    """Ground truth in the form of a label, output neuron
    readouts aim to produce at the end of each example"""
    def add_to_neuron(self, backward: bool, model: NeuronModel, 
                      shape, batch_size: int, example_timesteps: int):
        # Add variable, shared across neurons to hold true label for batch
        model.add_var("YTrue", "uint8_t", 0,
                      VarAccess.READ_ONLY_SHARED_NEURON, reset=False)

        # If backward pass is required, add second variable to 
        # hold the true label for the backward pass
        if backward:
            model.add_var("YTrueBack", "uint8_t", 0, 
                          VarAccess.READ_ONLY_SHARED_NEURON, reset=False)

    def push_to_device(self, genn_pop, y_true, shape, batch_size: int,
                       example_timesteps: int):
        # Check shape
        y_true = np.asarray(y_true)
        if y_true.ndim != 1 or len(y_true) > batch_size:
            raise RuntimeError(f"Length of target data for "
                               f"ExampleLabel prediction should "
                               f"be < {batch_size}")

        # Copy flattened y_true into view
        if batch_size == 1:
            genn_pop.vars["YTrue"].view[:] = y_true[:]
        else:
            genn_pop.vars["YTrue"].view[:len(y_true), 0] = y_true

        # Push YTrue to device
        genn_pop.vars["YTrue"].push_to_device()

    @property
    def backward_shared_neuron_var_reset(self) -> List[Tuple[str, str, str]]:
        """
        Gets resets for any shared neuron variables this ground truth adds
        """
        return [("YTrueBack", "uint8_t", "YTrue")]
