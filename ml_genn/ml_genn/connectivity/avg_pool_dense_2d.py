import numpy as np
from math import ceil

from .connectivity import Connectivity
from ..initializers import Wrapper
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from pygenn.genn_model import create_custom_init_var_snippet_class
from ..utils.connectivity import get_param_2d
from ..utils.value import is_value_array

from pygenn.genn_wrapper import (SynapseMatrixType_DENSE_INDIVIDUALG,
                                 SynapseMatrixType_DENSE_PROCEDURALG)

genn_snippet = create_custom_init_var_snippet_class(
    "avepool2d_dense",

    param_names=[
        "pool_kh", "pool_kw",
        "pool_sh", "pool_sw",
        "pool_ih", "pool_iw", "pool_ic",
        "dense_ih", "dense_iw", "dense_ic",
        "dense_units"],

    extra_global_params=[("weights", "scalar*")],

    var_init_code=
        """
        const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
        const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
        const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);

        // Convert presynaptic neuron ID to row, column and channel in pool input
        const int poolInRow = ($(id_pre) / pool_ic) / pool_iw;
        const int poolInCol = ($(id_pre) / pool_ic) % pool_iw;
        const int poolInChan = $(id_pre) % pool_ic;

        // Calculate corresponding pool output
        const int poolOutRow = poolInRow / pool_sh;
        const int poolStrideRow = poolOutRow * pool_sh;
        const int poolOutCol = poolInCol / pool_sw;
        const int poolStrideCol = poolOutCol * pool_sw;

        $(value) = 0.0;
        if ((poolInRow < (poolStrideRow + pool_kh)) && (poolInCol < (poolStrideCol + pool_kw))) {
            const int dense_ih = $(dense_ih), dense_iw = $(dense_iw), dense_ic = $(dense_ic);

            if ((poolOutRow < dense_ih) && (poolOutCol < dense_iw)) {
                const int dense_units = $(dense_units);
                const int dense_in_unit = poolOutRow * (dense_iw * dense_ic) + poolOutCol * (dense_ic) + poolInChan;
                const int dense_out_unit = $(id_post);

                $(value) = $(weights)[
                    dense_in_unit * (dense_units) +
                    dense_out_unit];
            }
        }
        """)


class AvgPoolDense2D(Connectivity):
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

    def connect(self, source, target):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.pool_output_shape = (
            ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
            ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
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

    def get_snippet(self, connection, supported_matrix_type):
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
            [SynapseMatrixType_DENSE_INDIVIDUALG, 
             SynapseMatrixType_DENSE_PROCEDURALG])

        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "AvgPoolDense2D connectivity")
        else:
            return ConnectivitySnippet(
                snippet=None, matrix_type=best_matrix_type,
                weight=wu_var_val, delay=self.delay)
