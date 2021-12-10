import numpy as np
from math import ceil
from pygenn.genn_model import (init_connectivity, init_toeplitz_connectivity,
                               init_var)

from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers.base_synapses import BaseSynapses
from ml_genn.layers.weight_update_models import signed_static_pulse
from ml_genn.layers.helper import _get_param_2d

class Conv2DSynapses(BaseSynapses):

    def __init__(self, filters, conv_size, conv_strides=None,
                 conv_padding='valid', connectivity_type='procedural'):
        super(Conv2DSynapses, self).__init__()
        self.filters = filters
        self.conv_size = _get_param_2d('conv_size', conv_size)
        self.conv_strides = _get_param_2d('conv_strides', conv_strides, default=(1, 1))
        self.conv_padding = PadMode(conv_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, source, target):
        super(Conv2DSynapses, self).connect(source, target)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = source.shape
        if self.conv_padding == PadMode.VALID:
            output_shape = (
                ceil(float(conv_ih - conv_kh + 1) / float(conv_sh)),
                ceil(float(conv_iw - conv_kw + 1) / float(conv_sw)),
                self.filters,
            )
        elif self.conv_padding == PadMode.SAME:
            output_shape = (
                ceil(float(conv_ih) / float(conv_sh)),
                ceil(float(conv_iw) / float(conv_sw)),
                self.filters,
            )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((conv_kh, conv_kw, conv_ic, self.filters), dtype=np.float64)

    def compile(self, mlg_model, name):
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.source().shape
        conv_oh, conv_ow, conv_oc = self.target().shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            # Calculate padding following approach described at
            # https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/nn_ops.py#L48-L88
            if (conv_ih % conv_sh == 0):
                conv_padh = max(conv_kh - conv_sh, 0) // 2
            else:
                conv_padh = max(conv_kh - (conv_ih % conv_sh), 0) // 2
            
            if (conv_iw % conv_sw == 0):
                conv_padw = max(conv_kw - conv_sw, 0) // 2
            else:
                conv_padw = max(conv_kw - (conv_iw % conv_sw), 0) // 2

        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'

        if self.connectivity_type == ConnectivityType.TOEPLITZ:
            assert conv_sh == 1 and conv_sw == 1
            conn_init = init_toeplitz_connectivity('Conv2D', {
                'conv_kh': conv_kh, 'conv_kw': conv_kw,
                #'conv_sh': conv_sh, 'conv_sw': conv_sw,
                'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
                'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc})
            conn = 'TOEPLITZ_KERNELG'
            wu_var = {'g': self.weights.flatten()}
            wu_var_egp = {}
        else:
            conn_init = init_connectivity('Conv2D', {
                'conv_kh': conv_kh, 'conv_kw': conv_kw,
                'conv_sh': conv_sh, 'conv_sw': conv_sw,
                'conv_padh': conv_padh, 'conv_padw': conv_padw,
                'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
                'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc})

            if self.connectivity_type == ConnectivityType.PROCEDURAL:
                conn = 'PROCEDURAL_KERNELG'
                wu_var = {'g': self.weights.flatten()}
                wu_var_egp = {}
            else:
                conn = 'SPARSE_INDIVIDUALG'
                wu_var = {'g': init_var('Kernel', {})}
                wu_var_egp = {'g': {'kernel': self.weights.flatten()}}

        super(Conv2DSynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                            {}, {}, 'DeltaCurr', {}, {}, conn_init, wu_var_egp)
