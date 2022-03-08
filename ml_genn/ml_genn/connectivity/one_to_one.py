from pygenn.genn_wrapper import (SynapseMatrixConnectivity_PROCEDURAL,
                                 SynapseMatrixConnectivity_SPARSE)
from .sparse_base import SparseBase
from ..utils import InitValue, Value

class OneToOne(SparseBase):
    def __init__(self, weight:InitValue, delay:InitValue=0):
        super(OneToOne, self).__init__(weight, delay)

    def connect(self, source, target):
        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("Target population shape mismatch")

    def get_snippet(self, connection, prefer_in_memory):
        return super(OneToOne, self)._get_snippet(prefer_in_memory, 
                                                  "OneToOne")
