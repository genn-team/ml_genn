import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel

from pygenn import VarAccess


class SparseCategoricalCrossentropy(Loss):
    """Computes the crossentropy between labels and prediction 
    when there are two or more label classes, specified as integers."""
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add variable, shared across neurons to hold true label for batch
        model.add_var("YTrue", "uint8_t", 0, 
                      VarAccess.READ_ONLY_SHARED_NEURON)

        # Add sim-code to convert label to one-hot
        model.append_sim_code(
            f"""
            const scalar yTrue = (id == YTrue) ? 1.0 : 0.0;
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
        genn_pop.vars["YTrue"].view[:len(y_true), 0] = y_true

        # Push YTrue to device
        genn_pop.vars["YTrue"].push_to_device()
