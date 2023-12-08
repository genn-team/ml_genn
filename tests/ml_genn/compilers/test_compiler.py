from ml_genn.compilers.compiler import SupportedMatrixType

from pytest import raises

from pygenn import SynapseMatrixType

def test_supported_matrix_type():
    supported_matrix_types = [SynapseMatrixType.DENSE,
                              SynapseMatrixType.SPARSE,
                              SynapseMatrixType.PROCEDURAL_KERNELG,
                              SynapseMatrixType.TOEPLITZ]
    
    a = SupportedMatrixType(supported_matrix_types)
    assert a.get_best([SynapseMatrixType.PROCEDURAL]) is None
    assert (a.get_best([SynapseMatrixType.SPARSE,
                        SynapseMatrixType.DENSE])
            == SynapseMatrixType.DENSE)