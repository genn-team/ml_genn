from .sparse_base import SparseBase
from ..utils.value import InitValue

from pygenn import init_connectivity


class OneToOne(SparseBase):
    def __init__(self, weight: InitValue, delay: InitValue = 0):
        super(OneToOne, self).__init__(weight, delay)

    def connect(self, source, target):
        output_shape = source.shape

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("Target population shape mismatch")

    def get_snippet(self, connection, supported_matrix_type):
        return super(OneToOne, self)._get_snippet(
            supported_matrix_type,
            init_connectivity("OneToOne"))
