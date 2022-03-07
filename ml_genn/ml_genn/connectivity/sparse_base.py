from pygenn.genn_wrapper import (SynapseMatrixType_PROCEDURAL_PROCEDURALG,
                                 SynapseMatrixType_SPARSE_INDIVIDUALG)
from .connectivity import Connectivity, Snippet
from ..utils import InitValue, Value

class SparseBase(Connectivity):
    def _get_snippet(self, prefer_in_memory, snippet):
        # If either weight or delay is an array or we prefer in-memory 
        # connectivity, only option is to use SPARSE_INDIVIDUALG
        if self.weight.is_array or self.delay.is_array or prefer_in_memory:
            return Snippet(snippet=snippet, 
                           matrix_type=SynapseMatrixType_SPARSE_INDIVIDUALG,
                           weight_var=self.weight, weight_var_egp=None,
                           delay_var=self.delay, delay_var_egp=None)
        # Otherwise, we can use PROCEDURAL_PROCEDURALG 
        # **NOTE** same as PROCEDURAL_GLOBALG for constant weights/delays
        else:
            return Snippet(snippet=snippet, 
                           matrix_type=SynapseMatrixType_PROCEDURAL_PROCEDURALG,
                           weight=self.weight, delay=self.delay)
    
