from pygenn.genn_wrapper import (SynapseMatrixConnectivity_PROCEDURAL,
                                 SynapseMatrixConnectivity_SPARSE)
from . import Connectivity
from ..utils import InitValue, Value

class OneToOne(Connectivity):
    def __init__(self, weight:InitValue, delay:InitValue=0):
        super(OneToOne, self).__init__(weight, delay)

    def connect(self, source, target):
        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("Target population shape mismatch")

    def get_snippet(self, prefer_in_memory):
        matrix_connectivity = (SynapseMatrixConnectivity_PROCEDURAL if prefer_in_memory
                               else SynapseMatrixConnectivity_SPARSE)
        return Snippet(conn_init=None, 
                       matrix_connectivity=matrix_connectivity)