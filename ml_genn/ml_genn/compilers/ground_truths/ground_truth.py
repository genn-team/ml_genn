from abc import ABC
from ml_genn.utils.model import NeuronModel

from abc import abstractmethod


class GroundTruth(ABC):
    """Base class for all ground truths"""
    @abstractmethod
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        """Modify a neuron model, adding any additional state 
        required to deliver this ground truth to device code.

        Args:
            model:              Neuron model to modify (in place) 
            shape:              Shape of population ground 
                                truth is provided for
            batch_size:         Batch size of model used with ground truth
            example_timesteps:  How many timestamps each example will be
                                presented to the network for
        """
        pass

    @abstractmethod
    def push_to_device(self, genn_pop, y_true, shape, batch_size: int,
                       example_timesteps: int):
        """
        Push the ground truth values for current example
        to the compiled neuron group.

        Args:
            genn_pop:           GeNN ``NeuronGroup`` ground truth is 
                                to be delivered to
            y_true:             Ground truth values for this example
            shape:              Shape of population ground 
                                truth is provided for
            batch_size:         Batch size of model used with ground truth
            example_timesteps:  How many timestamps each example will be
                                presented to the network for
        """
        pass
