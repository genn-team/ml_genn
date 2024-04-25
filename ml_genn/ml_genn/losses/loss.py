from abc import ABC
from ..utils.model import NeuronModel

from abc import abstractmethod


class Loss(ABC):
    """Base class for all loss functions"""
    @abstractmethod
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        """Modify a neuron model, adding any additional state 
        and functionality required to implement this loss function.

        Args:
            model:              Neuron model to modify (in place) 
            shape:              Shape of population loss is calculated for
            batch_size:         Batch size of model used with loss function
            example_timesteps:  How many timestamps each example will be
                                presented to the network for
        """
        pass

    @abstractmethod
    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        """
        Write the current target output value to the compiled neuron group.

        Args:
            genn_pop:           GeNN ``NeuronGroup`` object population with
                                loss function has been compiled into
            y_true:             'true' values provided to compiled network
                                train method
            shape:              Shape of population loss is calculated for
            batch_size:         Batch size of model used with loss function
            example_timesteps:  How many timestamps each example will be
                                presented to the network for
        """
        pass
