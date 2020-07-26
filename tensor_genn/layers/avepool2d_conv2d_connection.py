import numpy as np
from math import ceil

from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.base_connection import BaseConnection


avepool2d_conv2d_init = create_custom_init_var_snippet_class(
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

    extra_global_params=[
        ('kernels', 'scalar*'),
    ],

    var_init_code='''
    const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
    const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
    const int pool_padh = $(pool_padh), pool_padw = $(pool_padw);
    const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);

    const int pool_in_row = ($(id_pre) / pool_ic) / pool_iw;
    const int pool_in_col = ($(id_pre) / pool_ic) % pool_iw;
    const int pool_in_chan = $(id_pre) % pool_ic;

    const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);
    const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);
    const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);
    const int conv_ic = $(conv_ic);
    const int conv_ow = $(conv_ow), conv_oc = $(conv_oc);

    const int conv_out_row = ($(id_post) / conv_oc) / conv_ow;
    const int conv_out_col = ($(id_post) / conv_oc) % conv_ow;
    const int conv_out_chan = $(id_post) % conv_oc;

    int conv_stride_row = conv_out_row * conv_sh - conv_padh;
    int conv_stride_col = conv_out_col * conv_sw - conv_padw;

    scalar weight = 0.0;

    // process only strides with rows containing pool_in_row
    int pool_out_row = (pool_in_row + pool_padh) / pool_sh;
    int pool_stride_row = pool_out_row * pool_sh - pool_padh;
    while ((pool_stride_row >= -pool_padh) && (pool_stride_row + pool_kh > pool_in_row)) {

        int pool_kh_crop = min(pool_stride_row + pool_kh, pool_ih) - max(pool_stride_row, 0);

        // process only strides with cols containing pool_in_col
        int pool_out_col = (pool_in_col + pool_padw) / pool_sw;
        int pool_stride_col = pool_out_col * pool_sw - pool_padw;
        while ((pool_stride_col >= -pool_padw) && (pool_stride_col + pool_kw > pool_in_col)) {

            int pool_kw_crop = min(pool_stride_col + pool_kw, pool_iw) - max(pool_stride_col, 0);

            int conv_in_row = pool_out_row;
            int conv_in_col = pool_out_col;
            int conv_in_chan = pool_in_chan;

            int conv_k_row = conv_in_row - conv_stride_row;
            int conv_k_col = conv_in_col - conv_stride_col;

            if (conv_k_row >= 0 && conv_k_row < conv_kh && conv_k_col >= 0 && conv_k_col < conv_kw) {
                weight += $(kernels)[
                    conv_k_row * (conv_kw * conv_ic * conv_oc) +
                    conv_k_col * (conv_ic * conv_oc) +
                    conv_in_chan * (conv_oc) +
                    conv_out_chan
                ] / (pool_kh_crop * pool_kw_crop);
            }

            pool_out_col--;
            pool_stride_col = pool_out_col * pool_sw - pool_padw;
        }

        pool_out_row--;
        pool_stride_row = pool_out_row * pool_sh - pool_padh;
    }

    $(value) = weight;
    ''',
)


class AvePool2DConv2DConnection(BaseConnection):

    def __init__(self, filters, pool_size, conv_size, pool_strides=None, conv_strides=None,
                 pool_padding='valid', conv_padding='valid'):
        super(AvePool2DConv2DConnection, self).__init__()
        self.filters = filters
        self.pool_size = pool_size
        self.conv_size = conv_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.pool_padding = PadMode(pool_padding)
        self.conv_padding = PadMode(conv_padding)
        self.pool_output_shape = None
        self.conv_output_shape = None


    def compile(self, tg_model):
        super(AvePool2DConv2DConnection, self).compile(tg_model)

        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = self.source.shape
        if self.pool_padding == PadMode.VALID:
            pool_padh = 0
            pool_padw = 0
        elif self.pool_padding == PadMode.SAME:
            pool_padh = (pool_kh - 1) // 2
            pool_padw = (pool_kw - 1) // 2

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.pool_output_shape
        conv_oh, conv_ow, conv_oc = self.conv_output_shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = (conv_kh - 1) // 2
            conv_padw = (conv_kw - 1) // 2

        weights_init = init_var(avepool2d_conv2d_init, {
            'pool_kh': pool_kh, 'pool_kw': pool_kw,
            'pool_sh': pool_sh, 'pool_sw': pool_sw,
            'pool_padh': pool_padh, 'pool_padw': pool_padw,
            'pool_ih': pool_ih, 'pool_iw': pool_iw, 'pool_ic': pool_ic,
            'conv_kh': conv_kh, 'conv_kw': conv_kw,
            'conv_sh': conv_sh, 'conv_sw': conv_sw,
            'conv_padh': conv_padh, 'conv_padw': conv_padw,
            'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
            'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc,
        })

        for batch_i in range(tg_model.batch_size):
            pre_nrn = self.source.nrn[batch_i]
            post_nrn = self.target.nrn[batch_i]
            syn_name = '{}_to_{}_syn_{}'.format(self.source.name, self.target.name, batch_i)

            # Batch master synapses
            if not tg_model.share_weights or batch_i == 0:
                self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                    syn_name, 'DENSE_PROCEDURALG', NO_DELAY, pre_nrn, post_nrn,
                    'StaticPulse', {}, {'g': weights_init}, {}, {}, 'DeltaCurr', {}, {}
                )
                self.syn[batch_i].vars['g'].set_extra_global_init_param('kernels', self.weights.flatten())

            # Batch slave synapses
            else:
                master_syn_name = '{}_to_{}_syn_0'.format(self.source.name, self.target.name)
                self.syn[batch_i] = tg_model.g_model.add_slave_synapse_population(
                    syn_name, master_syn_name, NO_DELAY, pre_nrn, post_nrn, 'DeltaCurr', {}, {}
                )


    def connect(self, source, target):
        super(AvePool2DConv2DConnection, self).connect(source, target)

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
            self.conv_output_shape = (
                ceil(float(conv_ih - conv_kh + 1) / float(conv_sh)),
                ceil(float(conv_iw - conv_kw + 1) / float(conv_sw)),
                self.filters,
            )
        elif self.conv_padding == PadMode.SAME:
            self.conv_output_shape = (
                ceil(float(conv_ih) / float(conv_sh)),
                ceil(float(conv_iw) / float(conv_sw)),
                self.filters,
            )

        if target.shape is None:
            target.shape = self.conv_output_shape
        elif self.conv_output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((conv_kh, conv_kw, conv_ic, self.filters), dtype=np.float64)
