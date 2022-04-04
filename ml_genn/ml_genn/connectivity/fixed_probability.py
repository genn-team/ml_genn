from .sparse_base import SparseBase
from ..utils import InitValue

from pygenn.genn_model import init_connectivity

from ..utils import ConstantValueDescriptor

class FixedProbability(SparseBase):
    p = ConstantValueDescriptor()

    def __init__(self, p: float, weight:InitValue, allow_self_connections=False, delay:InitValue=0):
        super(FixedProbability, self).__init__(weight, delay)
        
        self.p = p
        self.allow_self_connections = allow_self_connections
    
    def connect(self, source, target):
        pass

    def get_snippet(self, connection, prefer_in_memory):
        # No autapse model should be used if self-connections are disallowed
        # and we're connecting the same population to itself
        no_autapse = (not self.allow_self_connections 
                      and connection.source() == connection.target())
        snippet = ("FixedProbabilityNoAutapse" if no_autapse 
                   else "FixedProbability")
        return super(FixedProbability, self)._get_snippet(
            prefer_in_memory, init_connectivity(snippet, {"prob": self.p}))
