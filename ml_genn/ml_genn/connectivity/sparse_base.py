from pygenn import SynapseMatrixType
from .connectivity import Connectivity
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue, ValueDescriptor

from ..utils.value import is_value_array


class SparseBase(Connectivity):
    """Base class for all sparse connectivity"""
    pre_ind = ValueDescriptor("pre_ind")
    post_ind = ValueDescriptor("post_ind")
    
    def __init__(self, weight: InitValue, delay: InitValue):
        super(SparseBase, self).__init__(weight, delay)
        
        # Initialise pre and postsynaptic indices
        self.pre_ind = None
        self.post_ind = None        

    def _get_snippet(self, supported_matrix_type, snippet):
        # Are pre or postsynaptic indices set?
        inds_provided = (self.pre_ind is not None 
                         or self.post_ind is not None)
        array_weight_delay = (is_value_array(self.weight)
                              or is_value_array(self.delay))
        
        # Build list of available matrix types, adding procedural
        # if indices aren't provided and weights aren't arrays
        # **NOTE** same as PROCEDURAL_GLOBALG for constant weights/delays
        available_matrix_types = [SynapseMatrixType.SPARSE]
        if not inds_provided and not array_weight_delay:
            available_matrix_types.append(SynapseMatrixType.PROCEDURAL)

        # Get best supported connectivity choice
        best_matrix_type = supported_matrix_type.get_best(
            available_matrix_types)
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "sparse connectivity")
        elif best_matrix_type == SynapseMatrixType.SPARSE:
            # If indices are provided
            if inds_provided:
                # If either of them are set to anything 
                # other than an array, give error
                if (not is_value_array(self.pre_ind) 
                    or not is_value_array(self.post_ind)):
                        raise RuntimeError("Pre and postsynaptic indices "
                                           "for manually specifying "
                                           "sparse connectivity must be "
                                           "provided as arrays")
                # Use sparse connectivity
                return ConnectivitySnippet(
                    snippet=None,
                    matrix_type=SynapseMatrixType.SPARSE,
                    weight=self.weight, delay=self.delay,
                    pre_ind=self.pre_ind, post_ind=self.post_ind)
                    
            # Otherwise, use snippet to initialize sparse matrix
            else:
                return ConnectivitySnippet(
                    snippet=snippet,
                    matrix_type=SynapseMatrixType.SPARSE,
                    weight=self.weight, delay=self.delay)
        # Otherwise, we can use PROCEDURAL_PROCEDURALG
        else:
            return ConnectivitySnippet(
                snippet=snippet,
                matrix_type=SynapseMatrixType.PROCEDURAL,
                weight=self.weight, delay=self.delay)
