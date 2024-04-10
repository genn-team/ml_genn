from __future__ import annotations

from math import ceil

from pygenn import SynapseMatrixType
from typing import Optional, TYPE_CHECKING
from .connectivity import Connectivity
from ..utils.connectivity import PadMode, Param2D, KernelInit
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType

from pygenn import (create_sparse_connect_init_snippet,
                    init_sparse_connectivity, init_toeplitz_connectivity)
from ..utils.connectivity import (get_conv_same_padding, get_param_2d,
                                  update_target_shape)
from ..utils.value import is_value_array


genn_snippet = create_sparse_connect_init_snippet(
    "avg_pool_conv_2d",

    params=[
        ("pool_kh", "int"), ("pool_kw", "int"),
        ("pool_sh", "int"), ("pool_sw", "int"),
        ("pool_ih", "int"), ("pool_iw", "int"), ("pool_ic", "int"),
        ("conv_kh", "int"), ("conv_kw", "int"),
        ("conv_sh", "int"), ("conv_sw", "int"),
        ("conv_padh", "int"), ("conv_padw", "int"),
        ("conv_ih", "int"), ("conv_iw", "int"), ("conv_ic", "int"),
        ("conv_oh", "int"), ("conv_ow", "int"), ("conv_oc", "int")],

    calc_max_row_len_func=lambda num_pre, num_post, pars: ceil(pars["conv_kh"] / pars["conv_sh"]) * ceil(pars["conv_kw"] / pars["conv_sw"]) * pars["conv_oc"],

    calc_kernel_size_func=lambda pars: [pars["conv_kh"], pars["conv_kw"], pars["conv_ic"], pars["conv_oc"]],

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
            if ((poolOutRow < conv_ih) && (poolOutCol < conv_iw)) {
                // Calculate range of output rows and columns which this pool output connects to
                const int minOutRow = min(conv_oh, max(0, 1 + (int)floor((poolOutRow + conv_padh - conv_kh) / (scalar)conv_sh)));
                const int maxOutRow = min(conv_oh, max(0, 1 + ((poolOutRow + conv_padh) / conv_sh)));
                const int minOutCol = min(conv_ow, max(0, 1 + (int)floor((poolOutCol + conv_padw - conv_kw) / (scalar)conv_sw)));
                const int maxOutCol = min(conv_ow, max(0, 1 + ((poolOutCol + conv_padw) / conv_sw)));

                // Loop through output rows, columns and channels
                for(int convOutRow = minOutRow; convOutRow < maxOutRow; convOutRow++) {
                    const int strideRow = (convOutRow * conv_sh) - conv_padh;
                    const int kernRow = poolOutRow - strideRow;
                    for(int convOutCol = minOutCol; convOutCol < maxOutCol; convOutCol++) {
                        const int strideCol = (convOutCol * conv_sw) - conv_padw;
                        const int kernCol = poolOutCol - strideCol;
                        for(int outChan = 0; outChan < conv_oc; outChan++) {
                            // Calculate postsynaptic index and add synapse
                            const int idPost = ((convOutRow * conv_ow * conv_oc) +
                                                (convOutCol * conv_oc) +
                                                outChan);
                            addSynapse(idPost, kernRow, kernCol, poolInChan, outChan);
                        }
                    }
                }
            }
        }
        """)


class AvgPoolConv2D(Connectivity):
    """Average pooling connectivity from source populations with 2D shape, 
    fused with convolution. These are typically used when converting ANNs
    where there is no non-linearity between Average Pooling and 
    Convolutional layers.
    
    Args:
        weight:         Convolution kernel weights. Must be either a constant
                        value, a :class:`ml_genn.initializers.Initializer` or
                        a numpy array whose shape matches ``conv_size`` 
                        and ``filters``.
        filters:        The number of filters in the convolution
        pool_size:      Factors by which to downscale. If only one integer
                        is specified, the same factor will be used 
                        for both dimensions.
        conv_size:      The size of the convolution window. If only one
                        integer is specified, the same factor will be used
                        for both dimensions.
        flatten:        Should shape of output be flattened?
        pool_strides:   Strides values for the pooling. These will default
                        to ``pool_size``. If only one integer is specified,
                        the same stride will be used for both dimensions.
        conv_strides:   Strides values for the convoltion. These will default
                        to ``(1, 1)``. If only one integer is specified, 
                        the same stride will be used for both dimensions.
        conv_padding:   either "valid" or "same". "valid" means no padding. 
                        "same" results in padding evenly to the left/right 
                        or up/down of the input. When padding="same" and 
                        strides=1, the output has the same size as the input.
        delay:          Homogeneous connection delays
    """
    def __init__(self, weight: InitValue, filters: int, pool_size: Param2D,
                 conv_size: Param2D, flatten: bool = False, 
                 pool_strides: Optional[Param2D] = None, 
                 conv_strides: Optional[Param2D] = None,
                 conv_padding: str = "valid", delay: InitValue = 0):
        super(AvgPoolConv2D, self).__init__(weight, delay)
        self.filters = filters
        self.pool_size = get_param_2d("pool_size", pool_size)
        self.conv_size = get_param_2d("conv_size", conv_size)
        self.flatten = flatten
        self.pool_strides = get_param_2d("pool_strides", pool_strides,
                                         default=self.pool_size)
        self.conv_strides = get_param_2d("conv_strides", conv_strides,
                                         default=(1, 1))
        self.conv_padding = PadMode(conv_padding)
        self.pool_output_shape = None

        if (self.pool_strides[0] < self.pool_size[0]
                or self.pool_strides[1] < self.pool_size[1]):
            raise NotImplementedError("pool stride < pool size "
                                      "is not supported")
        if self.conv_strides[0] != 1 or self.conv_strides[1] != 1:
            raise NotImplementedError("conv stride != 1 is not supported")

    def connect(self, source: Population, target: Population):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.pool_output_shape = (
            ceil((pool_ih - pool_kh + 1) / pool_sh),
            ceil((pool_iw - pool_kw + 1) / pool_sw),
            pool_ic)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.pool_output_shape
        if self.conv_padding == PadMode.VALID:
            self.output_shape = (
                ceil((conv_ih - conv_kh + 1) / conv_sh),
                ceil((conv_iw - conv_kw + 1) / conv_sw),
                self.filters)
        elif self.conv_padding == PadMode.SAME:
            self.output_shape = (ceil(conv_ih / conv_sh),
                                 ceil(conv_iw / conv_sw),
                                 self.filters)

        # Update target shape
        update_target_shape(target, self.output_shape, self.flatten)

        # Check shape of weights matches kernels
        weight_shape = (conv_kh, conv_kw, conv_ic, self.filters)
        if is_value_array(self.weight) and self.weight.shape != weight_shape:
            raise RuntimeError("If weights are specified as arrays, they "
                               "should  match shape of AvgPoolConv2D kernel")

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = connection.source().shape
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.pool_output_shape
        conv_oh, conv_ow, conv_oc = self.output_shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = get_conv_same_padding(conv_ih, conv_kh, conv_sh)
            conv_padw = get_conv_same_padding(conv_iw, conv_kw, conv_sw)

        scaled_weight = self.weight.flatten() / (pool_kh * pool_kw)

        # Build list of available matrix types, 
        # adding Toeplitz of constraints are met
        available_matrix_types = [SynapseMatrixType.SPARSE,
                                  SynapseMatrixType.PROCEDURAL_KERNELG]
        if (conv_sh == 1 and conv_sw == 1
            and (self.conv_padding is not PadMode.SAME
                 or ((self.conv_size[0] % 2) != 0
                     and (self.conv_size[1] % 2) != 0))):
            available_matrix_types.append(SynapseMatrixType.TOEPLITZ)

        # Get best supported connectivity choice
        best_matrix_type = supported_matrix_type.get_best(
            available_matrix_types)
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "AvgPoolConv2D connectivity")
        elif best_matrix_type == SynapseMatrixType.TOEPLITZ:
            conn_init = init_toeplitz_connectivity("AvgPoolConv2D", {
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "pool_kh": pool_kh, "pool_kw": pool_kw,
                "pool_sh": pool_sh, "pool_sw": pool_sw,
                "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            return ConnectivitySnippet(
                snippet=conn_init,
                matrix_type=SynapseMatrixType.TOEPLITZ,
                weight=scaled_weight, delay=self.delay)
        else:
            conn_init = init_sparse_connectivity(genn_snippet, {
                "pool_kh": pool_kh, "pool_kw": pool_kw,
                "pool_sh": pool_sh, "pool_sw": pool_sw,
                "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "conv_sh": conv_sh, "conv_sw": conv_sw,
                "conv_padh": conv_padh, "conv_padw": conv_padw,
                "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            if best_matrix_type == SynapseMatrixType.PROCEDURAL_KERNELG:
                return ConnectivitySnippet(
                    snippet=conn_init,
                    matrix_type=SynapseMatrixType.PROCEDURAL_KERNELG,
                    weight=scaled_weight,
                                           delay=self.delay)
            else:
                # If weights/delays are arrays, use kernel initializer
                # to initialize, otherwise use as is
                weight = (KernelInit(scaled_weight)
                          if is_value_array(self.weight)
                          else scaled_weight)
                delay = (KernelInit(self.delay)
                         if is_value_array(self.delay)
                         else self.delay)
                return ConnectivitySnippet(
                    snippet=conn_init,
                    matrix_type=SynapseMatrixType.SPARSE,
                    weight=weight, delay=delay)
