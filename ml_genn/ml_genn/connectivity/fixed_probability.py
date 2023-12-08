from .sparse_base import SparseBase
from ..utils.snippet import ConstantValueDescriptor
from ..utils.value import InitValue

from pygenn import init_sparse_connectivity


class FixedProbability(SparseBase):
    p = ConstantValueDescriptor()

    def __init__(self, p: float, weight: InitValue,
                 allow_self_connections=False, delay: InitValue = 0):
        super(FixedProbability, self).__init__(weight, delay)

        self.p = p
        self.allow_self_connections = allow_self_connections

    def connect(self, source, target):
        pass

    def get_snippet(self, connection, supported_matrix_type):
        # No autapse model should be used if self-connections are disallowed
        # and we're connecting the same population to itself
        no_autapse = (not self.allow_self_connections
                      and connection.source() == connection.target())
        snippet = ("FixedProbabilityNoAutapse" if no_autapse
                   else "FixedProbability")
        return super(FixedProbability, self)._get_snippet(
            supported_matrix_type, 
            init_sparse_connectivity(snippet, {"prob": self.p}))
