import numpy as np

from .connectivity import Connectivity, Snippet
from ..utils import InitValue, Value

class Dense(Connectivity):
    def __init__(self, weight:InitValue, delay:InitValue=0):
        super(Dense, self).__init__(weight, delay)

    def connect(self, source, target):
        # If weights are specified as 2D array
        if self.weight.is_array and self.weight.value.ndim == 2:
            source_size, target_size = self.weight.value.shape
            
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
        # **TODO** should we do something sensible with 1D arrays too
    
    def get_snippet(self, connection, prefer_in_memory):
        return Snippet(snippet=None, 
                       matrix_type="DENSE_INDIVIDUALG",
                       weight=self.weight, delay=self.delay)
        
