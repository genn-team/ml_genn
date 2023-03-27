from math import ceil

from .connectivity import Connectivity
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from pygenn.genn_model import (create_cmlf_class,
                               create_custom_sparse_connect_init_snippet_class,
                               init_connectivity)
from ..utils.connectivity import get_param_2d, update_target_shape
from ..utils.value import is_value_constant

from pygenn.genn_wrapper import (SynapseMatrixType_SPARSE_INDIVIDUALG,
                                 SynapseMatrixType_PROCEDURAL_PROCEDURALG)
                                 
genn_snippet = create_custom_sparse_connect_init_snippet_class(
    "avg_pool_2d",

    param_names=[
        "pool_kh", "pool_kw",
        "pool_sh", "pool_sw",
        "pool_ih", "pool_iw", "pool_ic",
        "pool_oh", "pool_ow", "pool_oc"],

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
    def __init__(self, pool_size, flatten=False,
                 pool_strides=None, delay: InitValue = 0):
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

    def connect(self, source, target):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.output_shape = (
            ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
            ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
            pool_ic)

        # Update target shape
        update_target_shape(target, self.output_shape, self.flatten)

    def get_snippet(self, connection, supported_matrix_type):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = connection.source().shape
        pool_oh, pool_ow, pool_oc = self.output_shape

        conn_init = init_connectivity(genn_snippet, {
            "pool_kh": pool_kh, "pool_kw": pool_kw,
            "pool_sh": pool_sh, "pool_sw": pool_sw,
            "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
            "pool_oh": pool_oh, "pool_ow": pool_ow, "pool_oc": pool_oc})
        
        # Get best supported matrix type
        # **NOTE** no need to use globalg as constant weights 
        # will be turned into parameters which is equivalent
        best_matrix_type = supported_matrix_type.get_best(
            [SynapseMatrixType_SPARSE_INDIVIDUALG, 
             SynapseMatrixType_PROCEDURAL_PROCEDURALG])

        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "AvgPool2D connectivity")
        else:
            return ConnectivitySnippet(snippet=conn_init,
                                       matrix_type=best_conn,
                                       weight=self.weight, delay=self.delay,
                                       trainable=False)
