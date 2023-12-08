from .connectivity import Connectivity
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from ..utils.value import is_value_array

from pygenn import SynapseMatrixType

class Dense(Connectivity):
    def __init__(self, weight: InitValue, delay: InitValue = 0):
        super(Dense, self).__init__(weight, delay)

    def connect(self, source, target):
        # If weights are specified as 2D array
        if is_value_array(self.weight):
            if self.weight.ndim != 2:
                raise NotImplementedError("Dense connectivity requires "
                                          "a 2D array of weights")

            source_size, target_size = self.weight.shape

            # Set/check target shape
            if target.shape is None:
                target.shape = (target_size,)
            elif target.shape != (target_size,):
                raise RuntimeError("target population shape "
                                   "doesn't match weights")

            # Set/check source shape
            if source.shape is None:
                source.shape = (source_size,)
            elif source.shape != (source_size,):
                raise RuntimeError("source population shape "
                                   "doesn't match weights")

    def get_snippet(self, connection, supported_matrix_type):
        # Get best supported connectivity choice
        best_matrix_type = supported_matrix_type.get_best(
            [SynapseMatrixType.DENSE])
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "Dense connectivity")
        else:
            return ConnectivitySnippet(
                snippet=None,
                matrix_type=SynapseMatrixType.DENSE,
                weight=self.weight, delay=self.delay)
