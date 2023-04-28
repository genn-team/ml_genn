from ml_genn.compilers.compiler import SupportedMatrixType

from pytest import raises

from pygenn.genn_wrapper import (SynapseMatrixType_DENSE_INDIVIDUALG,
                                 SynapseMatrixType_SPARSE_INDIVIDUALG,
                                 SynapseMatrixType_PROCEDURAL_KERNELG,
                                 SynapseMatrixType_PROCEDURAL_PROCEDURALG,
                                 SynapseMatrixType_TOEPLITZ_KERNELG)

def test_supported_matrix_type():
    supported_matrix_types = [SynapseMatrixType_DENSE_INDIVIDUALG,
                              SynapseMatrixType_SPARSE_INDIVIDUALG,
                              SynapseMatrixType_PROCEDURAL_KERNELG,
                              SynapseMatrixType_TOEPLITZ_KERNELG]
    
    a = SupportedMatrixType(supported_matrix_types)
    assert a.get_best([SynapseMatrixType_PROCEDURAL_PROCEDURALG]) is None
    assert (a.get_best([SynapseMatrixType_SPARSE_INDIVIDUALG,
                        SynapseMatrixType_DENSE_INDIVIDUALG])
            == SynapseMatrixType_DENSE_INDIVIDUALG)