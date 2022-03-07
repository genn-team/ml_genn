import numpy as np
from math import ceil

from pygenn.genn_wrapper import (SynapseMatrixConnectivity_PROCEDURAL,
                                 SynapseMatrixConnectivity_SPARSE,
                                 SynapseMatrixConnectivity_TOEPLITZ)
from . import Connectivity
from .helper import PadMode
from ..utils import InitValue, Value

from pygenn.genn_model import (init_connectivity, init_toeplitz_connectivity, 
                               init_var)
from .helper import _get_conv_same_padding, _get_param_2d

class Conv2D(Connectivity):
    def __init__(self, weight:InitValue, filters, conv_size, conv_strides=None,
                 conv_padding="valid", delay:InitValue=0):
        super(Conv2D, self).__init__(weight, delay)

        self.filters = filters
        self.conv_size = _get_param_2d("conv_size", conv_size)
        self.conv_strides = _get_param_2d("conv_strides", conv_strides, default=(1, 1))
        self.conv_padding = PadMode(conv_padding)

    def connect(self, source, target):
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = source.shape
        if self.conv_padding == PadMode.VALID:
            output_shape = (
                ceil(float(conv_ih - conv_kh + 1) / float(conv_sh)),
                ceil(float(conv_iw - conv_kw + 1) / float(conv_sw)),
                self.filters)
        elif self.conv_padding == PadMode.SAME:
            output_shape = (
                ceil(float(conv_ih) / float(conv_sh)),
                ceil(float(conv_iw) / float(conv_sw)),
                self.filters)

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError("target layer shape mismatch")

        # Check shape of weights matches kernels
        weight_shape = (conv_kh, conv_kw, conv_ic, self.filters)
        if (isinstance(self.weight, (Sequence, np.ndarray)) 
                and self.weights.shape != weight_shape):
            raise RuntimeError("If weights are specified as arrays, they "
                               "should  match shape of Conv2D kernel")
    
    def get_snippet(self, prefer_in_memory):
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.source().shape
        conv_oh, conv_ow, conv_oc = self.target().shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = _get_conv_same_padding(conv_ih, conv_kh, conv_sh)
            conv_padw = _get_conv_same_padding(conv_iw, conv_kw, conv_sw)

        if prefer_in_memory and conv_sh == 1 and conv_sw == 1:
            conn_init = init_toeplitz_connectivity("Conv2D", {
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})
            return Snippet(conn_init=conn_init, 
                           matrix_connectivity=SynapseMatrixConnectivity_TOEPLITZ)
        else:
            conn_init = init_connectivity("Conv2D", {
                "conv_kh": conv_kh, "conv_kw": conv_kw,
                "conv_sh": conv_sh, "conv_sw": conv_sw,
                "conv_padh": conv_padh, "conv_padw": conv_padw,
                "conv_ih": conv_ih, "conv_iw": conv_iw, "conv_ic": conv_ic,
                "conv_oh": conv_oh, "conv_ow": conv_ow, "conv_oc": conv_oc})

            if prefer_in_memory:
                return Snippet(conn_init=conn_init, 
                               matrix_connectivity=SynapseMatrixConnectivity_PROCEDURAL)
                
            else:
                return Snippet(conn_init=conn_init, 
                               matrix_connectivity=SynapseMatrixConnectivity_SPARSE)
                
        
