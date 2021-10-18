import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_sparse_connect_init_snippet_class
from pygenn.genn_model import (init_connectivity, init_var, 
                               create_cmlf_class, create_cksf_class)
from pygenn.genn_wrapper.StlContainers import UnsignedIntVector

from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers.base_synapses import BaseSynapses
from ml_genn.layers.weight_update_models import signed_static_pulse
from ml_genn.layers.helper import _get_param_2d

avepool2d_conv2d_init = create_custom_sparse_connect_init_snippet_class(
    'avepool2d_conv2d',

    param_names=[
        'pool_kh', 'pool_kw',
        'pool_sh', 'pool_sw',
        'pool_padh', 'pool_padw',
        'pool_ih', 'pool_iw', 'pool_ic',
        'conv_kh', 'conv_kw',
        'conv_sh', 'conv_sw',
        'conv_padh', 'conv_padw',
        'conv_ih', 'conv_iw', 'conv_ic',
        'conv_oh', 'conv_ow', 'conv_oc',
    ],

    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: (int(pars[9]) // int(pars[11])) * (int(pars[10]) // int(pars[12])) * int(pars[20]))(),

    calc_kernel_size_func=create_cksf_class(
        lambda pars: UnsignedIntVector([int(pars[9]), int(pars[10]), int(pars[17]), int(pars[20])]))(),

    row_build_code='''
    // Stash all parameters in registers
    // **NOTE** this means parameters from group structure only get converted from float->int once
    // **NOTE** if they're actually constant, compiler is still likely to treat them as constants rather than allocating registers
    const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
    const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
    const int pool_padh = $(pool_padh), pool_padw = $(pool_padw);
    const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);
    const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);
    const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);
    const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);
    const int conv_ow = $(conv_ow), conv_oh = $(conv_oh), conv_oc = $(conv_oc);

    // Convert presynaptic neuron ID to row, column and channel in pool input
    const int poolInRow = ($(id_pre) / pool_ic) / pool_iw;
    const int poolInCol = ($(id_pre) / pool_ic) % pool_iw;
    const int poolInChan = $(id_pre) % pool_ic;

    // Calculate corresponding pool output
    const int poolOutRow = (poolInRow + pool_padh) / pool_sh;
    const int poolStrideRow = poolOutRow * pool_sh - pool_padh;
    const int poolCropKH = min(poolStrideRow + pool_kh, pool_ih) - max(poolStrideRow, 0);
    const int poolOutCol = (poolInCol + pool_padw) / pool_sw;
    const int poolStrideCol = poolOutCol * pool_sw - pool_padw;
    const int poolCropKW = min(poolStrideCol + pool_kw, pool_iw) - max(poolStrideCol, 0);

    if ((poolInRow < (poolStrideRow + pool_kh)) && (poolInCol < (poolStrideCol + pool_kw))) {

        // Calculate range of output rows and columns which this pool output connects to
        const int minOutRow = min(conv_oh, max(0, 1 + ((poolOutRow + conv_padh - conv_kh) / conv_sh)));
        const int maxOutRow = min(conv_oh, max(0, 1 + ((poolOutRow + conv_padh) / conv_sh)));
        const int minOutCol = min(conv_ow, max(0, 1 + ((poolOutCol + conv_padw - conv_kw) / conv_sw)));
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

    // End the row
    $(endRow);
    ''',
)

class AvePool2DConv2DSynapses(BaseSynapses):

    def __init__(self, filters, pool_size, conv_size, pool_strides=None, 
                 conv_strides=None, pool_padding='valid', 
                 conv_padding='valid', connectivity_type='procedural'):
        super(AvePool2DConv2DSynapses, self).__init__()
        self.filters = filters
        self.pool_size = _get_param_2d('pool_size', pool_size)
        self.conv_size = _get_param_2d('conv_size', conv_size)
        self.pool_strides = _get_param_2d('pool_strides', pool_strides, default=self.pool_size)
        self.conv_strides = _get_param_2d('conv_strides', conv_strides, default=(1, 1))
        self.pool_padding = PadMode(pool_padding)
        self.conv_padding = PadMode(conv_padding)
        self.pool_output_shape = None
        self.connectivity_type = ConnectivityType(connectivity_type)
        if self.pool_strides[0] < self.pool_size[0] or self.pool_strides[1] < self.pool_size[1]:
            raise NotImplementedError('pool stride < pool size is not supported')

    def connect(self, source, target):
        super(AvePool2DConv2DSynapses, self).connect(source, target)

        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        if self.pool_padding == PadMode.VALID:
            self.pool_output_shape = (
                ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
                ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
                pool_ic,
            )
        elif self.pool_padding == PadMode.SAME:
            self.pool_output_shape = (
                ceil(float(pool_ih) / float(pool_sh)),
                ceil(float(pool_iw) / float(pool_sw)),
                pool_ic,
            )

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.pool_output_shape
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
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = self.source().shape
        if self.pool_padding == PadMode.VALID:
            pool_padh = 0
            pool_padw = 0
        elif self.pool_padding == PadMode.SAME:
            pool_padh = (pool_kh - 1) // 2
            pool_padw = (pool_kw - 1) // 2

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.pool_output_shape
        conv_oh, conv_ow, conv_oc = self.target().shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = (conv_kh - 1) // 2
            conv_padw = (conv_kw - 1) // 2

        conn_init = init_connectivity(avepool2d_conv2d_init, {
            'pool_kh': pool_kh, 'pool_kw': pool_kw,
            'pool_sh': pool_sh, 'pool_sw': pool_sw,
            'pool_padh': pool_padh, 'pool_padw': pool_padw,
            'pool_ih': pool_ih, 'pool_iw': pool_iw, 'pool_ic': pool_ic,
            'conv_kh': conv_kh, 'conv_kw': conv_kw,
            'conv_sh': conv_sh, 'conv_sw': conv_sw,
            'conv_padh': conv_padh, 'conv_padw': conv_padw,
            'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
            'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc})

        conn = ('PROCEDURAL_PROCEDURALG' if self.connectivity_type == ConnectivityType.PROCEDURAL
                else 'SPARSE_INDIVIDUALG')
        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': init_var('Kernel', {})}
        wu_var_egp = {'g': {'kernel': self.weights.flatten() / (pool_kh * pool_kw)}}

        super(AvePool2DConv2DSynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                                     {}, {}, 'DeltaCurr', {}, {}, conn_init, wu_var_egp)
