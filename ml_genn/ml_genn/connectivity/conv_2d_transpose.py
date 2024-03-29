from pygenn.genn_wrapper.StlContainers import UnsignedIntVector
from .connectivity import Connectivity
from ..utils.connectivity import PadMode, KernelInit
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from pygenn.genn_model import (create_cksf_class, create_cmlf_class,
                               create_custom_sparse_connect_init_snippet_class,
                               init_connectivity)
from ..utils.connectivity import (get_conv_same_padding, get_param_2d,
                                  update_target_shape)
from ..utils.value import is_value_array

from pygenn.genn_wrapper import (SynapseMatrixType_SPARSE_INDIVIDUALG,
                                 SynapseMatrixType_PROCEDURAL_KERNELG)

genn_snippet = create_custom_sparse_connect_init_snippet_class(
    "conv_2d_transpose",

    param_names=["conv_kh", "conv_kw",
                 "conv_sh", "conv_sw",
                 "conv_padh", "conv_padw",
                 "conv_ih", "conv_iw", "conv_ic",
                 "conv_oh", "conv_ow", "conv_oc"],

    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[0] * pars[1] * pars[11]))(),

    calc_kernel_size_func=create_cksf_class(
        lambda pars: UnsignedIntVector([int(pars[0]), int(pars[1]),
                                        int(pars[11]), int(pars[8])]))(),

    row_build_state_vars=[
        ("inRow", "int", "($(id_pre) / (int) $(conv_ic)) / (int) $(conv_iw)"),
        ("inCol", "int", "($(id_pre) / (int) $(conv_ic)) % (int) $(conv_iw)"),
        ("inChan", "int", "$(id_pre) % (int) $(conv_ic)"),
        ("strideRow", "int", "(inRow * (int) $(conv_sh)) - (int) $(conv_padh)"),
        ("strideCol", "int", "(inCol * (int) $(conv_sw)) - (int) $(conv_padw)"),
        ("outRow", "int", "min((int) $(conv_oh), max(0, (inRow * (int) $(conv_sh)) - (int) $(conv_padh)))"),
        ("maxOutRow", "int", "min((int) $(conv_oh), max(0, (inRow * (int) $(conv_sh)) + (int) $(conv_kh) - (int) $(conv_padh)))"),
        ("minOutCol", "int", "min((int) $(conv_ow), max(0, (inCol * (int) $(conv_sw)) - (int) $(conv_padw)))"),
        ("maxOutCol", "int", "min((int) $(conv_ow), max(0, (inCol * (int) $(conv_sw)) + (int) $(conv_kw) - (int) $(conv_padw)))")],

    row_build_code=
        """
        if ($(outRow) == $(maxOutRow)) {
           $(endRow);
        }
        const int kernRow = $(outRow) - $(strideRow);
        for (int outCol = $(minOutCol); outCol < $(maxOutCol); outCol++) {
            const int kernCol = outCol - $(strideCol);
            for (unsigned int outChan = 0; outChan < (unsigned int) $(conv_oc); outChan++) {
                const int idPost = (($(outRow) * (int) $(conv_ow) * (int) $(conv_oc)) +
                                    (outCol * (int) $(conv_oc)) +
                                    outChan);
                $(addSynapse, idPost, kernRow, kernCol, outChan, inChan);
            }
        }
        $(outRow)++;
        """)


class Conv2DTranspose(Connectivity):
    def __init__(self, weight: InitValue, filters, conv_size,
                 flatten=False, conv_strides=None, 
                 conv_padding="valid", delay: InitValue = 0):
        super(Conv2DTranspose, self).__init__(weight, delay)

        self.filters = filters
        self.conv_size = get_param_2d("conv_size", conv_size)
        self.flatten = flatten
        self.conv_strides = get_param_2d("conv_strides", conv_strides,
                                         default=(1, 1))
        self.conv_padding = PadMode(conv_padding)

    def connect(self, source, target):
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

    def get_snippet(self, connection, supported_matrix_type):
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

        conn_init = init_connectivity(genn_snippet, {
            "conv_kh": conv_kh, "conv_kw": conv_kw,
            "conv_sh": conv_sh, "conv_sw": conv_sw,
            "conv_padh": conv_padh, "conv_padw": conv_padw,
            "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
            "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

        # Get best supported connectivity choice
        best_matrix_type = supported_matrix_type.get_best(
            [SynapseMatrixType_SPARSE_INDIVIDUALG,
             SynapseMatrixType_PROCEDURAL_KERNELG])
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "Conv2DTranspose connectivity")
        elif best_matrix_type == SynapseMatrixType_SPARSE_INDIVIDUALG:
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
                matrix_type=SynapseMatrixType_SPARSE_INDIVIDUALG,
                weight=weight, delay=delay)
        else:
            return ConnectivitySnippet(
                snippet=conn_init,
                matrix_type=SynapseMatrixType_PROCEDURAL_KERNELG,
                weight=self.weight,
                delay=self.delay)
