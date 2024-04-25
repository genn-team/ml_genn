from __future__ import annotations

from abc import ABC

from abc import abstractmethod, abstractproperty

class Communicator(ABC):
    """Base class for all communicators"""
    @abstractmethod
    def barrier(self):
        """Wait for all ranks to reach this point in execution before continuing
        """
        pass
    
    @abstractmethod
    def broadcast(self, data, root: int):
        """Broadcast data from root to all ranks
        Args:
            data: Data to broadcast
            root: Index of node to broadcast from
        """
        pass

    @abstractmethod
    def reduce_sum(self, value):
        """Calculates the sum of value across all ranks
        Args:
            value: Value to sum up
        Returns:
            Sum of value across all ranks
        """
        pass

    @abstractproperty
    def rank(self):
        """Gets index of this rank

        Returns:
            int: rank
        """
        pass

    @abstractproperty
    def num_ranks(self):
        """Gets total number of ranks

        Returns:
            int: num_ranks
        """
        pass
