import numpy as np

from pygenn.genn_wrapper import SynapseMatrixConnectivity_DENSE
from . import Connectivity, Snippet
from ..utils import InitValue, Value

class Dense(Connectivity):
    def __init__(self, weight:InitValue, delay:InitValue=0):
        super(Dense, self).__init__(weight, delay)

    def connect(self, source, target):
        """
        output_shape = (self.units, )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')
        """
    
    def get_snippet(self, prefer_in_memory):
        return Snippet(conn_init=None, 
                       matrix_connectivity=SynapseMatrixConnectivity_DENSE)
