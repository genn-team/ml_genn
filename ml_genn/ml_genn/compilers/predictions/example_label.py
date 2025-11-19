import numpy as np

from .prediction import Prediction
from ..utils.model import NeuronModel

from pygenn import VarAccess


class ExampleLabel(Prediction):
    """Prediction in the form of a label, output neuron
    readouts need to produce at the end of each example"""
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add variable, shared across neurons to hold true label for batch
        model.add_var("YTrue", "uint8_t", 0, 
                      VarAccess.READ_ONLY_SHARED_NEURON)


    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
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
