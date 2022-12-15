from .connectivity import Connectivity
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue, ValueDescriptor

from ..utils.value import is_value_array


class SparseBase(Connectivity):
    pre_ind = ValueDescriptor("pre_ind")
    post_ind = ValueDescriptor("post_ind")
    
    def __init__(self, weight: InitValue, delay: InitValue):
        super(SparseBase, self).__init__(weight, delay)
        
        # Initialise pre and postsynaptic indices
        self.pre_ind = None
        self.post_ind = None        

    def _get_snippet(self, prefer_in_memory, snippet):
        # If either pre or postsynaptic indices are set
        if self.pre_ind is not None or self.post_ind is not None:
            # If either of them are set to anything 
            # other than an array, give error
            if (not is_value_array(self.pre_ind) 
                or not is_value_array(self.post_ind)):
                    raise RuntimeError("Pre and postsynaptic indices "
                                       "for manually specifying "
                                       "sparse connectivity must be "
                                       "provided as arrays")
            # Use sparse connectivity
            return ConnectivitySnippet(snippet=None,
                                       matrix_type="SPARSE_INDIVIDUALG",
                                       weight=self.weight, delay=self.delay,
                                       pre_ind=self.pre_ind,
                                       post_ind=self.post_ind)
                
        # Otherwise, if either weight or delay is an array or we prefer in-memory
        # connectivity, only option is to use SPARSE_INDIVIDUALG
        elif (is_value_array(self.weight) or is_value_array(self.delay)
                or prefer_in_memory):
            return ConnectivitySnippet(snippet=snippet,
                                       matrix_type="SPARSE_INDIVIDUALG",
                                       weight=self.weight, delay=self.delay)
        # Otherwise, we can use PROCEDURAL_PROCEDURALG
        # **NOTE** same as PROCEDURAL_GLOBALG for constant weights/delays
        else:
            return ConnectivitySnippet(snippet=snippet,
                                       matrix_type="PROCEDURAL_PROCEDURALG",
                                       weight=self.weight, delay=self.delay)
