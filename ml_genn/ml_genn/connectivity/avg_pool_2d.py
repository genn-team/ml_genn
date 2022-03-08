import numpy as np
from math import ceil

from pygenn.genn_wrapper import (SynapseMatrixType_PROCEDURAL_GLOBALG,
                                 SynapseMatrixType_SPARSE_GLOBALG)
from .connectivity import Connectivity, KernelInit
from .helper import PadMode
from ..utils import InitValue, Value

from pygenn.genn_model import (create_cmlf_class,
                               create_custom_sparse_connect_init_snippet_class, 
                               init_connectivity, init_var)
from .helper import _get_conv_same_padding, _get_param_2d

genn_snippet = create_custom_sparse_connect_init_snippet_class(
    "avg_pool_2d",

    param_names=[
        "pool_kh", "pool_kw",
        "pool_sh", "pool_sw",
        "pool_ih", "pool_iw", "pool_ic",
        "pool_oh", "pool_ow", "pool_oc",
    ],

    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(ceil(pars[0] / pars[2])) * int(ceil(pars[1] / pars[3])) * int(pars[9]))(),

    row_build_code=
        """
        // Stash all parameters in registers
        // **NOTE** this means parameters from group structure only get converted from float->int once
        // **NOTE** if they"re actually constant, compiler is still likely to treat them as constants rather than allocating registers
        const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
        const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
        const int pool_iw = $(pool_iw), pool_ic = $(pool_ic);
        const int pool_oh = $(pool_oh), pool_ow = $(pool_ow), pool_oc = $(pool_oc);
        
        // Convert presynaptic neuron ID to row, column and channel in pool input
        const int poolInRow = ($(id_pre) / pool_ic) / pool_iw;
        const int poolInCol = ($(id_pre) / pool_ic) % pool_iw;
        const int poolInChan = $(id_pre) % pool_ic;
        
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
                $(addSynapse, idPost);
            }
        }
        // End the row
        $(endRow);
        """)

class AvgPool2D(Connectivity):
    def __init__(self, pool_size, pool_strides=None, delay:InitValue=0):
        self.pool_size = _get_param_2d("pool_size", pool_size)
        self.pool_strides = _get_param_2d("pool_strides", pool_strides, default=self.pool_size)
        
        super(AvgPool2D, self).__init__(1.0 / (self.pool_size[0] * self.pool_size[1]), delay)
        
        if self.pool_strides[0] < self.pool_size[0] or self.pool_strides[1] < self.pool_size[1]:
            raise NotImplementedError("pool stride < pool size is not supported")
        
        if not self.delay.is_constant:
            raise NotImplementedError("AvgPool2D connectivity only supports constant delays")

    def connect(self, source, target):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        output_shape = (
            ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
            ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
            pool_ic)
        
        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("target layer shape mismatch")

    def get_snippet(self, connection, prefer_in_memory):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = connection.source.shape
        pool_oh, pool_ow, pool_oc = connection.target.shape

        conn_init = init_connectivity(genn_snippet, {
            "pool_kh": pool_kh, "pool_kw": pool_kw,
            "pool_sh": pool_sh, "pool_sw": pool_sw,
            "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
            "pool_oh": pool_oh, "pool_ow": pool_ow, "pool_oc": pool_oc})
        
        if prefer_in_memory:
            return Snippet(snippet=conn_init, 
                            matrix_type=SynapseMatrixType_SPARSE_GLOBALG,
                            weight=self.weight, delay=self.delay)
        else:
            return Snippet(snippet=conn_init, 
                            matrix_type=SynapseMatrixType_PROCEDURAL_GLOBALG,
                            weight=self.weight, delay=self.delay)
            
