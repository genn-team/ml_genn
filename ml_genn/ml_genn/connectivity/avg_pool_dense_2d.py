from __future__ import annotations

import numpy as np
from math import ceil

from pygenn import SynapseMatrixType
from typing import TYPE_CHECKING
from .connectivity import Connectivity
from ..initializers import Wrapper
from ..utils.connectivity import Param2D
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType

from pygenn import create_var_init_snippet
from ..utils.connectivity import get_param_2d
from ..utils.value import is_value_array


genn_snippet = create_var_init_snippet(
    "avepool2d_dense",

    params=[
        ("pool_kh", "int"), ("pool_kw", "int"),
        ("pool_sh", "int"), ("pool_sw", "int"),
        ("pool_ih", "int"), ("pool_iw", "int"), ("pool_ic", "int"),
        ("dense_ih", "int"), ("dense_iw", "int"), ("dense_ic", "int"),
        ("dense_units", "int")],

    extra_global_params=[("weights", "scalar*")],

    var_init_code=
        """
        // Convert presynaptic neuron ID to row, column and channel in pool input
        const int poolInRow = (id_pre / pool_ic) / pool_iw;
        const int poolInCol = (id_pre / pool_ic) % pool_iw;
        const int poolInChan = id_pre % pool_ic;

        // Calculate corresponding pool output
        const int poolOutRow = poolInRow / pool_sh;
        const int poolStrideRow = poolOutRow * pool_sh;
        const int poolOutCol = poolInCol / pool_sw;
        const int poolStrideCol = poolOutCol * pool_sw;

        value = 0.0;
        if ((poolInRow < (poolStrideRow + pool_kh)) && (poolInCol < (poolStrideCol + pool_kw))) {
            if ((poolOutRow < dense_ih) && (poolOutCol < dense_iw)) {
                const int dense_in_unit = poolOutRow * (dense_iw * dense_ic) + poolOutCol * (dense_ic) + poolInChan;

                value = weights[
                    dense_in_unit * (dense_units) +
                    id_post];
            }
        }
        """)


class AvgPoolDense2D(Connectivity):
    """Average pooling connectivity from source populations with 2D shape, 
    fused with dense layer. These are typically used when converting ANNs
    where there is no non-linearity between Average Pooling and 
    dense layers.
    
    Args:
        weight:         Connection weights
        pool_size:      Factors by which to downscale. If only one integer
                        is specified, the same factor will be used 
                        for both dimensions.
        pool_strides:   Strides values for the pooling. These will default
                        to ``pool_size``. If only one integer is specified,
                        the same stride will be used for both dimensions.
        delay:          Homogeneous connection delays
    """
    def __init__(self, weight: InitValue, pool_size, pool_strides=None,
                 delay: InitValue = 0):
        super(AvgPoolDense2D, self).__init__(weight, delay)

        self.pool_size = get_param_2d("pool_size", pool_size)
        self.pool_strides = get_param_2d("pool_strides", pool_strides,
                                         default=self.pool_size)
        self.pool_output_shape = None

        if (self.pool_strides[0] < self.pool_size[0]
                or self.pool_strides[1] < self.pool_size[1]):
            raise NotImplementedError("pool stride < pool size "
                                      "is not supported")

    def connect(self, source: Population, target: Population):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.pool_output_shape = (
            ceil((pool_ih - pool_kh + 1) / pool_sh),
            ceil((pool_iw - pool_kw + 1) / pool_sw),
            pool_ic)

        # If weights are specified as 2D array
        if is_value_array(self.weight):
            if self.weight.ndim != 2:
                raise NotImplementedError("AvgPoolDense2D connectivity "
                                          "requires a 2D array of weights")

            source_size, target_size = self.weight.shape

            # Set/check target shape
            if target.shape is None:
                target.shape = (target_size,)
            elif target.shape != (target_size,):
                raise RuntimeError("target population shape "
                                   "doesn't match weights")

            # Set/check source shape
            if np.prod(self.pool_output_shape) != source_size:
                raise RuntimeError("pool output size doesn't "
                                   "match weights")

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = connection.source().shape
        dense_ih, dense_iw, dense_ic = self.pool_output_shape

        wu_var_val = Wrapper(genn_snippet, {
            "pool_kh": pool_kh, "pool_kw": pool_kw,
            "pool_sh": pool_sh, "pool_sw": pool_sw,
            "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
            "dense_ih": dense_ih, "dense_iw": dense_iw, "dense_ic": dense_ic,
            "dense_units": int(np.prod(connection.target().shape))},
            {"weights": self.weight.flatten() / (pool_kh * pool_kw)})

        # Get best supported matrix type
        best_matrix_type = supported_matrix_type.get_best(
            [SynapseMatrixType.DENSE,
             SynapseMatrixType.DENSE_PROCEDURALG])

        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "AvgPoolDense2D connectivity")
        else:
            return ConnectivitySnippet(
                snippet=None, matrix_type=best_matrix_type,
                weight=wu_var_val, delay=self.delay)
