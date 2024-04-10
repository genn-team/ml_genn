from __future__ import annotations

from pygenn import SynapseMatrixType
from typing import TYPE_CHECKING
from .connectivity import Connectivity
from ..utils.connectivity import PadMode, Param2D, KernelInit
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType

from pygenn import (create_sparse_connect_init_snippet, init_sparse_connectivity)
from ..utils.connectivity import (get_conv_same_padding, get_param_2d,
                                  update_target_shape)
from ..utils.value import is_value_array


genn_snippet = create_sparse_connect_init_snippet(
    "conv_2d_transpose",

    params=[("conv_kh", "int"), ("conv_kw", "int"),
            ("conv_sh", "int"), ("conv_sw", "int"),
            ("conv_padh", "int"), ("conv_padw", "int"),
            ("conv_ih", "int"), ("conv_iw", "int"), ("conv_ic", "int"),
            ("conv_oh", "int"), ("conv_ow", "int"), ("conv_oc", "int")],

    calc_max_row_len_func=lambda num_pre, num_post, pars: int(pars["conv_kh"] * pars["conv_kw"] * pars["conv_oc"]),

    calc_kernel_size_func=lambda pars: [pars["conv_kh"], pars["conv_kw"],
                                        pars["conv_oc"], pars["conv_ic"]],

    row_build_code=
        """
        const int inRow = (id_pre / conv_ic) / conv_iw;
        const int inCol = (id_pre / conv_ic) % conv_iw;
        const int inChan = id_pre % conv_ic;
        const int strideRow = (inRow * conv_sh) - conv_padh;
        const int strideCol = (inCol * conv_sw) - conv_padw;
        const int maxOutRow = min(conv_oh, max(0, (inRow * conv_sh) + conv_kh - conv_padh));
        const int minOutCol = min(conv_ow, max(0, (inCol * conv_sw) - conv_padw));
        const int maxOutCol = min(conv_ow, max(0, (inCol * conv_sw) + conv_kw - conv_padw));

        int outRow = min(conv_oh, max(0, (inRow * conv_sh) - conv_padh));
        for(;outRow < maxOutRow; outRow++) {
            const int kernRow = outRow - strideRow;
            for (int outCol = minOutCol; outCol < maxOutCol; outCol++) {
                const int kernCol = outCol - strideCol;
                for (int outChan = 0; outChan < conv_oc; outChan++) {
                    const int idPost = ((outRow * conv_ow * conv_oc) +
                                        (outCol * conv_oc) +
                                        outChan);
                    addSynapse(idPost, kernRow, kernCol, outChan, inChan);
                }
            }
        }
        """)


class Conv2DTranspose(Connectivity):
    """Transposed convolutional connectivity from source populations with 2D shape.
    
    Args:
        weight:         Convolution kernel weights. Must be either a constant
                        value, a :class:`ml_genn.initializers.Initializer` or
                        a numpy array whose shape matches ``conv_size`` 
                        and ``filters``.
        filters:        The number of filters in the convolution
        conv_size:      The size of the convolution window. If only one
                        integer is specified, the same factor will be used
                        for both dimensions.
        flatten:        Should shape of output be flattened?
        conv_strides:   Strides values for the convoltion. These will default
                        to ``(1, 1)``. If only one integer is specified, 
                        the same stride will be used for both dimensions.
        conv_padding:   either "valid" or "same". "valid" means no padding. 
                        "same" results in padding evenly to the left/right 
                        or up/down of the input. When padding="same" and 
                        strides=1, the output has the same size as the input.
        delay:          Homogeneous connection delays
    """
    def __init__(self, weight: InitValue, filters: int, 
                 conv_size: Param2D, flatten=False, 
                 conv_strides: Optional[Param2D] = None, 
                 conv_padding: str = "valid", delay: InitValue = 0):
        super(Conv2DTranspose, self).__init__(weight, delay)

        self.filters = filters
        self.conv_size = get_param_2d("conv_size", conv_size)
        self.flatten = flatten
        self.conv_strides = get_param_2d("conv_strides", conv_strides,
                                         default=(1, 1))
        self.conv_padding = PadMode(conv_padding)

    def connect(self, source: Population, target: Population):
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = source.shape
        if self.conv_padding is PadMode.VALID:
            self.output_shape = (
                conv_ih * conv_sh + max(conv_kh - conv_sh, 0),
                conv_iw * conv_sw + max(conv_kw - conv_sw, 0),
                self.filters)
        elif self.conv_padding is PadMode.SAME:
            self.output_shape = (conv_ih * conv_sh,
                                 conv_iw * conv_sw,
                                 self.filters)

        # Update target shape
        update_target_shape(target, self.output_shape, self.flatten)

        # Check shape of weights matches kernels
        weight_shape = (conv_kh, conv_kw, self.filters, conv_ic)
        if is_value_array(self.weight) and self.weight.shape != weight_shape:
            raise RuntimeError("If weights are specified as arrays, they "
                               "should match shape of Conv2DTranspose kernel")

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = connection.source().shape
        conv_oh, conv_ow, conv_oc = self.output_shape
        if self.conv_padding is PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding is PadMode.SAME:
            conv_padh = get_conv_same_padding(conv_ih, conv_kh, conv_sh)
            conv_padw = get_conv_same_padding(conv_iw, conv_kw, conv_sw)

        conn_init = init_sparse_connectivity(genn_snippet, {
            "conv_kh": conv_kh, "conv_kw": conv_kw,
            "conv_sh": conv_sh, "conv_sw": conv_sw,
            "conv_padh": conv_padh, "conv_padw": conv_padw,
            "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
            "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

        # Get best supported connectivity choice
        best_matrix_type = supported_matrix_type.get_best(
            [SynapseMatrixType.SPARSE,
             SynapseMatrixType.PROCEDURAL_KERNELG])
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "Conv2DTranspose connectivity")
        elif best_matrix_type == SynapseMatrixType.SPARSE:
            # If weights/delays are arrays, use kernel initializer
            # to initialize, otherwise use as is
            weight = (KernelInit(self.weight)
                      if is_value_array(self.weight)
                      else self.weight)
            delay = (KernelInit(self.delay)
                     if is_value_array(self.delay)
                     else self.delay)
            return ConnectivitySnippet(
                snippet=conn_init,
                matrix_type=SynapseMatrixType.SPARSE,
                weight=weight, delay=delay)
        else:
            return ConnectivitySnippet(
                snippet=conn_init,
                matrix_type=SynapseMatrixType.PROCEDURAL_KERNELG,
                weight=self.weight,
                delay=self.delay)
