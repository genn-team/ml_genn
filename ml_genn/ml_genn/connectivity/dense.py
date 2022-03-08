import numpy as np

from pygenn.genn_wrapper import SynapseMatrixType_DENSE_INDIVIDUALG
from .connectivity import Connectivity, Snippet
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
    
    def get_snippet(self, connection, prefer_in_memory):
        return Snippet(snippet=None, 
                       matrix_type=SynapseMatrixType_DENSE_INDIVIDUALG,
                       weight=self.weight, delay=self.delay)
        
