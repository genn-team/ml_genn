from __future__ import annotations

from typing import TYPE_CHECKING
from .sparse_base import SparseBase
from ..utils.snippet import ConstantValueDescriptor
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType

from pygenn import init_sparse_connectivity


class FixedProbability(SparseBase):
    """Sparse connectivity where there is a fixed probability of any 
    pair of source and target neurons being connected.
    
    Args:
        p:                      Probability of connection
        weight:                 Connection weights. Must be either a constant
                                value or a 
                                :class:`ml_genn.initializers.Initializer`
        allow_self_connections: If this connectivity is used to connect the
                                same population to itself, should the same 
                                neuron be allowed to connect to itself?
        delay:                  Connection delays
    """
    p = ConstantValueDescriptor()

    def __init__(self, p: float, weight: InitValue,
                 allow_self_connections: bool = False, delay: InitValue = 0):
        super(FixedProbability, self).__init__(weight, delay)

        self.p = p
        self.allow_self_connections = allow_self_connections

    def connect(self, source: Population, target: Population):
        pass

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        # No autapse model should be used if self-connections are disallowed
        # and we're connecting the same population to itself
        no_autapse = (not self.allow_self_connections
                      and connection.source() == connection.target())
        snippet = ("FixedProbabilityNoAutapse" if no_autapse
                   else "FixedProbability")
        return super(FixedProbability, self)._get_snippet(
            supported_matrix_type, 
            init_sparse_connectivity(snippet, {"prob": self.p}))
