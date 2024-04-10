from __future__ import annotations

import numpy as np

from abc import ABC
from typing import TYPE_CHECKING
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue, ValueDescriptor

from abc import abstractmethod
from ..utils.value import is_value_array

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType


class Connectivity(ABC):
    """Base class for all connectivity classes.
    
    Args:
        weight: Connection weights
        delay:  Connection delays
    """
    weight = ValueDescriptor("g")
    delay = ValueDescriptor("d")

    def __init__(self, weight: InitValue, delay: InitValue):
        self.weight = weight
        self.delay = delay

        # If both weight and delay are arrays, check they are the same shape
        weight_array = is_value_array(self.weight)
        delay_array = is_value_array(self.delay)
        if (weight_array and delay_array
                and np.shape(weight_array) != np.shape(delay_array)):
            raise RuntimeError("If weights and delays are specified as "
                               "arrays, they should be the same shape")

    @abstractmethod
    def connect(self, source: Population, target: Population):
        """Called when two populations are connected to validate 
        and potentially configure their shapes based on connectivity.
        
        Args:
            source: Source population
            target: Target population
        """
        pass

    @abstractmethod
    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        """Gets PyGeNN implementation of connectivity initializer

        Args:
            connection:             Connection this connectivity
                                    is to be applied to
            supported_matrix_type:  GeNN synaptic matrix datatypes 
                                    supported by current compiler
        """
        pass
