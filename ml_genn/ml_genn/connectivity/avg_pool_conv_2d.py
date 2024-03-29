from math import ceil

from pygenn.genn_wrapper.StlContainers import UnsignedIntVector
from .connectivity import Connectivity
from ..utils.snippet import ConnectivitySnippet
from ..utils.connectivity import PadMode, KernelInit
from ..utils.value import InitValue

from pygenn.genn_model import (create_cmlf_class, create_cksf_class,
                               create_custom_sparse_connect_init_snippet_class,
                               init_connectivity, init_toeplitz_connectivity)
from ..utils.connectivity import (get_conv_same_padding, get_param_2d,
                                  update_target_shape)
from ..utils.value import is_value_array

from pygenn.genn_wrapper import (SynapseMatrixType_SPARSE_INDIVIDUALG,
                                 SynapseMatrixType_PROCEDURAL_KERNELG,
                                 SynapseMatrixType_TOEPLITZ_KERNELG)

genn_snippet = create_custom_sparse_connect_init_snippet_class(
    "avg_pool_conv_2d",

    param_names=[
        "pool_kh", "pool_kw",
        "pool_sh", "pool_sw",
        "pool_ih", "pool_iw", "pool_ic",
        "conv_kh", "conv_kw",
        "conv_sh", "conv_sw",
        "conv_padh", "conv_padw",
        "conv_ih", "conv_iw", "conv_ic",
        "conv_oh", "conv_ow", "conv_oc"],

    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(ceil(pars[7] / pars[9])) * int(ceil(pars[8] / pars[10])) * int(pars[18]))(),

    calc_kernel_size_func=create_cksf_class(
        lambda pars: UnsignedIntVector([int(pars[7]), int(pars[8]), 
                                        int(pars[15]), int(pars[18])]))(),

        row_build_code=
        """
        // Stash all parameters in registers
        // **NOTE** this means parameters from group structure only get converted from float->int once
        // **NOTE** if they"re actually constant, compiler is still likely to treat them as constants rather than allocating registers
        const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
        const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
        const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);
        const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);
        const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);
        const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);
        const int conv_ih = $(conv_ih), conv_iw = $(conv_iw), conv_ic = $(conv_ic);
        const int conv_oh = $(conv_oh), conv_ow = $(conv_ow), conv_oc = $(conv_oc);

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
            if ((poolOutRow < conv_ih) && (poolOutCol < conv_iw)) {
                // Calculate range of output rows and columns which this pool output connects to
                const int minOutRow = min((int) $(conv_oh), max(0, 1 + (int) floor((poolOutRow + $(conv_padh) - $(conv_kh)) / $(conv_sh))));
                const int maxOutRow = min(conv_oh, max(0, 1 + ((poolOutRow + conv_padh) / conv_sh)));
                const int minOutCol = min((int) $(conv_ow), max(0, 1 + (int) floor((poolOutCol + $(conv_padw) - $(conv_kw)) / $(conv_sw))));
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
                            $(addSynapse, idPost, kernRow, kernCol, poolInChan, outChan);
                        }
                    }
                }
            }
        }

        // End the row
        $(endRow);
        """)


class AvgPoolConv2D(Connectivity):
    def __init__(self, weight: InitValue, filters: int, pool_size, conv_size,
                 flatten=False, pool_strides=None, conv_strides=None,
                 conv_padding="valid", delay: InitValue = 0):
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

    def connect(self, source, target):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.pool_output_shape = (
            ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
            ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
            pool_ic)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.pool_output_shape
        if self.conv_padding == PadMode.VALID:
            self.output_shape = (
                ceil(float(conv_ih - conv_kh + 1) / float(conv_sh)),
                ceil(float(conv_iw - conv_kw + 1) / float(conv_sw)),
                self.filters)
        elif self.conv_padding == PadMode.SAME:
            self.output_shape = (ceil(float(conv_ih) / float(conv_sh)),
                                 ceil(float(conv_iw) / float(conv_sw)),
                                 self.filters)

        # Update target shape
        update_target_shape(target, self.output_shape, self.flatten)

        # Check shape of weights matches kernels
        weight_shape = (conv_kh, conv_kw, conv_ic, self.filters)
        if is_value_array(self.weight) and self.weight.shape != weight_shape:
            raise RuntimeError("If weights are specified as arrays, they "
                               "should  match shape of AvgPoolConv2D kernel")

    def get_snippet(self, connection, supported_matrix_type):
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
        available_matrix_types = [SynapseMatrixType_SPARSE_INDIVIDUALG,
                                  SynapseMatrixType_PROCEDURAL_KERNELG]
        if (conv_sh == 1 and conv_sw == 1
            and (self.conv_padding is not PadMode.SAME
                 or ((self.conv_size[0] % 2) != 0
                     and (self.conv_size[1] % 2) != 0))):
            available_matrix_types.append(SynapseMatrixType_TOEPLITZ_KERNELG)

        # Get best supported connectivity choice
        best_matrix_type = supported_matrix_type.get_best(
            available_matrix_types)
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "AvgPoolConv2D connectivity")
        elif best_matrix_type == SynapseMatrixType_TOEPLITZ_KERNELG:
            conn_init = init_toeplitz_connectivity("AvgPoolConv2D", {
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "pool_kh": pool_kh, "pool_kw": pool_kw,
                "pool_sh": pool_sh, "pool_sw": pool_sw,
                "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            return ConnectivitySnippet(
                snippet=conn_init,
                matrix_type=SynapseMatrixType_TOEPLITZ_KERNELG,
                weight=scaled_weight, delay=self.delay)
        else:
            conn_init = init_connectivity(genn_snippet, {
                "pool_kh": pool_kh, "pool_kw": pool_kw,
                "pool_sh": pool_sh, "pool_sw": pool_sw,
                "pool_ih": pool_ih, "pool_iw": pool_iw, "pool_ic": pool_ic,
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "conv_sh": conv_sh, "conv_sw": conv_sw,
                "conv_padh": conv_padh, "conv_padw": conv_padw,
                "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            if best_matrix_type == SynapseMatrixType_PROCEDURAL_KERNELG:
                return ConnectivitySnippet(
                    snippet=conn_init,
                    matrix_type=SynapseMatrixType_PROCEDURAL_KERNELG,
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
                    matrix_type=SynapseMatrixType_SPARSE_INDIVIDUALG,
                    weight=weight, delay=delay)
