import numpy as np
from math import ceil

from .connectivity import Connectivity
from .helper import PadMode
from ..initializers import Wrapper
from ..utils import ConnectivitySnippet, InitValue, Value

from pygenn.genn_model import (create_custom_init_var_snippet_class)
from .helper import _get_param_2d

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
    def __init__(self, weight:InitValue, pool_size, pool_strides=None, delay:InitValue=0):
        super(AvgPoolDense2D, self).__init__(weight, delay)

        self.pool_size = _get_param_2d("pool_size", pool_size)
        self.pool_strides = _get_param_2d("pool_strides", pool_strides, default=self.pool_size)
        self.pool_output_shape = None

        if self.pool_strides[0] < self.pool_size[0] or self.pool_strides[1] < self.pool_size[1]:
            raise NotImplementedError("pool stride < pool size is not supported")

    def connect(self, source, target):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.pool_output_shape = (
            ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
            ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
            pool_ic)

    def get_snippet(self, connection, prefer_in_memory):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = connection.source().shape
        dense_ih, dense_iw, dense_ic = self.pool_output_shape

        conn = ("DENSE_INDIVIDUALG" if prefer_in_memory 
                else "DENSE_PROCEDURALG")
        
        wu_var_val = Wrapper(genn_snippet, {
            "pool_kh": pool_kh, "pool_kw": pool_kw,
            "pool_sh": pool_sh, "pool_sw": pool_sw,
            "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
            "dense_ih": dense_ih, "dense_iw": dense_iw, "dense_ic": dense_ic,
            "dense_units": int(np.prod(connection.target().shape))},
            {"weights": self.weight.value.flatten() / (pool_kh * pool_kw)})

        return ConnectivitySnippet(snippet=None, 
                                   matrix_type=conn,
                                   weight=Value(wu_var_val), 
                                   delay=self.delay)

