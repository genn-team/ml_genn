from __future__ import annotations

from typing import TYPE_CHECKING
from .sparse_base import SparseBase
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType

from pygenn import init_sparse_connectivity


class OneToOne(SparseBase):
    """Sparse connectivity where each neuron in the source population
    is connected to the neuron with the same index in the target 
    population (which must have the same size).
    
    Args:
        weight: Connection weights. Must be either a constant value, a
                :class:`ml_genn.initializers.Initializer` or a sequence
                containing a weight for each neuron
        delay:  Connection delays
    """
    def __init__(self, weight: InitValue, delay: InitValue = 0):
        super(OneToOne, self).__init__(weight, delay)

    def connect(self, source: Population, target: Population):
        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("Target population shape mismatch")

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        return super(OneToOne, self)._get_snippet(
            supported_matrix_type,
            init_sparse_connectivity("OneToOne"))
