from .connectivity import Connectivity
from ..utils import ConnectivitySnippet, InitValue, Value

from ..utils.value import is_value_array

class SparseBase(Connectivity):
    def _get_snippet(self, prefer_in_memory, snippet):
        # If either weight or delay is an array or we prefer in-memory 
        # connectivity, only option is to use SPARSE_INDIVIDUALG
        if (is_value_array(self.weight) or is_value_array(self.delay) 
                or prefer_in_memory):
            return Snippet(snippet=snippet, 
                           matrix_type="SPARSE_INDIVIDUALG",
                           weight=self.weight, delay=self.delay)
        # Otherwise, we can use PROCEDURAL_PROCEDURALG 
        # **NOTE** same as PROCEDURAL_GLOBALG for constant weights/delays
        else:
            return ConnectivitySnippet(snippet=snippet, 
                                       matrix_type="PROCEDURAL_PROCEDURALG",
                                       weight=self.weight, delay=self.delay)
    
