import numpy as np

from .ground_truth import GroundTruth
from ml_genn.utils.model import NeuronModel


class ExampleValue(GroundTruth):
    """Ground truth in the form of a value, output neuron
    readouts aim to produce at the end of each example"""
    def add_to_neuron(self, backward: bool, model: NeuronModel, 
                      shape, batch_size: int, example_timesteps: int):
        # Add variable, shared across neurons to hold true label for batch
        model.add_var("YTrue", "scalar", 0.0, 
                      VarAccess.READ_ONLY_DUPLICATE, reset=False)
        
        # If backward pass is required, add second variable to 
        # hold the true label for the backward pass
        if backward:
            model.add_var("YTrueBack", "scalar", 0.0,
                          VarAccess.READ_ONLY_DUPLICATE, reset=False)
        # Add sim-code to convert label to one-hot
        # **THINK** pointless
        #model.append_sim_code(
        #    f"""
        #    const scalar yTrue = YTrue;
        #    """)

    def push_to_device(self, genn_pop, y_true, shape, batch_size: int,
                       example_timesteps: int):
        # Check shape
        expected_shape = (batch_size,) + shape
        y_true = np.asarray(y_true)
        if y_true.shape != expected_shape:
            raise RuntimeError(f"Shape of target data for ExampleValue "
                               f"prediction should be {expected_shape}")

        # Copy flattened y_true into view
        genn_pop.vars["YTrue"].view[:batch_size, :] = y_true

        # Push YTrue to device
        genn_pop.vars["YTrue"].push_to_device()