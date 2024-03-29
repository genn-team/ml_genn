from math import ceil

from .connectivity import Connectivity
from ..utils.connectivity import PadMode, KernelInit
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from pygenn.genn_model import (init_connectivity, init_toeplitz_connectivity)
from ..utils.connectivity import (get_conv_same_padding, get_param_2d,
                                  update_target_shape)
from ..utils.value import is_value_array

from pygenn.genn_wrapper import (SynapseMatrixType_SPARSE_INDIVIDUALG,
                                 SynapseMatrixType_PROCEDURAL_KERNELG,
                                 SynapseMatrixType_TOEPLITZ_KERNELG)


class Conv2D(Connectivity):
    def __init__(self, weight: InitValue, filters, conv_size,
                 flatten=False, conv_strides=None,
                 conv_padding="valid", delay: InitValue = 0):
        super(Conv2D, self).__init__(weight, delay)

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
                               "should  match shape of Conv2D kernel")

    def get_snippet(self, connection, supported_matrix_type):
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = connection.source().shape
        conv_oh, conv_ow, conv_oc = self.output_shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = get_conv_same_padding(conv_ih, conv_kh, conv_sh)
            conv_padw = get_conv_same_padding(conv_iw, conv_kw, conv_sw)
        
        # Build list of available matrix types, 
        # adding Toeplitz of constraints are met
        available_matrix_types = [SynapseMatrixType_SPARSE_INDIVIDUALG,
                                  SynapseMatrixType_PROCEDURAL_KERNELG]
        if conv_sh == 1 and conv_sw == 1:
            available_matrix_types.append(SynapseMatrixType_TOEPLITZ_KERNELG)
        
        # Get best supported matrix type
        best_matrix_type = supported_matrix_type.get_best(
            available_matrix_types)
        
        if best_matrix_type is None:
            raise NotImplementedError("Compiler does not support "
                                      "Conv2D connectivity")
        elif best_matrix_type == SynapseMatrixType_TOEPLITZ_KERNELG:
            conn_init = init_toeplitz_connectivity("Conv2D", {
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            return ConnectivitySnippet(
                snippet=conn_init,
                matrix_type=SynapseMatrixType_TOEPLITZ_KERNELG,
                weight=self.weight, delay=self.delay)
        else:
            conn_init = init_connectivity("Conv2D", {
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "conv_sh": conv_sh, "conv_sw": conv_sw,
                "conv_padh": conv_padh, "conv_padw": conv_padw,
                "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            if best_matrix_type == SynapseMatrixType_SPARSE_INDIVIDUALG:
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
