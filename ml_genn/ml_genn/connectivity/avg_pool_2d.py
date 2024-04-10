from __future__ import annotations

from math import ceil

from pygenn import SynapseMatrixType
from typing import Optional, TYPE_CHECKING
from .connectivity import Connectivity
from ..utils.connectivity import Param2D
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType
    
from pygenn import (create_sparse_connect_init_snippet,
                    init_sparse_connectivity)
from ..utils.connectivity import get_param_2d, update_target_shape
from ..utils.value import is_value_constant


genn_snippet = create_sparse_connect_init_snippet(
    "avg_pool_2d",

    params=[
        ("pool_kh", "int"), ("pool_kw", "int"),
        ("pool_sh", "int"), ("pool_sw", "int"),
        ("pool_ih", "int"), ("pool_iw", "int"), ("pool_ic", "int"),
        ("pool_oh", "int"), ("pool_ow", "int"), ("pool_oc", "int")],

    calc_max_row_len_func=lambda num_pre, num_post, pars: ceil(pars["pool_kh"] / pars["pool_sh"]) * ceil(pars["pool_kw"] / pars["pool_sw"]) * pars["pool_oc"],

    row_build_code=
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

        if ((poolInRow < (poolStrideRow + pool_kh)) && (poolInCol < (poolStrideCol + pool_kw))) {
            if ((poolOutRow < pool_oh) && (poolOutCol < pool_ow)) {
                // Calculate postsynaptic index and add synapse
                const int idPost = ((poolOutRow * pool_ow * pool_oc) +
                                    (poolOutCol * pool_oc) +
                                    poolInChan);
                addSynapse(idPost);
            }
        }
        """)


class AvgPool2D(Connectivity):
    """Average pooling connectivity from source populations with 2D shape
    
    Args:
        pool_size:      Factors by which to downscale. If only one integer
                        is specified, the same factor will be used 
                        for both dimensions.
        flatten:        Should shape of output be flattened?
        pool_strides:   Strides values. These will default to ``pool_size``.
                        If only one integer is specified, the same stride 
                        will be used for both dimensions.
        delay:          Homogeneous connection delays
    """
    def __init__(self, pool_size: Param2D, flatten: bool = False,
                 pool_strides: Optional[Param2D] = None, 
                 delay: InitValue = 0):
        self.pool_size = get_param_2d("pool_size", pool_size)
        self.flatten = flatten
        self.pool_strides = get_param_2d("pool_strides", pool_strides,
                                         default=self.pool_size)

        super(AvgPool2D, self).__init__(
            1.0 / (self.pool_size[0] * self.pool_size[1]), delay)

        if (self.pool_strides[0] < self.pool_size[0]
                or self.pool_strides[1] < self.pool_size[1]):
            raise NotImplementedError("pool stride < pool size "
                                      "is not supported")

        if not is_value_constant(self.delay):
            raise NotImplementedError("AvgPool2D connectivity only "
                                      "supports constant delays")

    def connect(self, source: Population, target: Population):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.output_shape = (
            ceil((pool_ih - pool_kh + 1) / pool_sh),
            ceil((pool_iw - pool_kw + 1) / pool_sw),
            pool_ic)

        # Update target shape
        update_target_shape(target, self.output_shape, self.flatten)

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = connection.source().shape
        pool_oh, pool_ow, pool_oc = self.output_shape

        conn_init = init_sparse_connectivity(genn_snippet, {
            "pool_kh": pool_kh, "pool_kw": pool_kw,
            "pool_sh": pool_sh, "pool_sw": pool_sw,
            "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
            "pool_oh": pool_oh, "pool_ow": pool_ow, "pool_oc": pool_oc})

        # Get best supported matrix type
        # **NOTE** no need to use globalg as constant weights 
        # will be turned into parameters which is equivalent
        best_matrix_type = supported_matrix_type.get_best(
            [SynapseMatrixType.SPARSE, SynapseMatrixType.PROCEDURAL])

        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "AvgPool2D connectivity")
        else:
            return ConnectivitySnippet(snippet=conn_init,
                                       matrix_type=best_matrix_type,
                                       weight=self.weight, delay=self.delay,
                                       trainable=False)
