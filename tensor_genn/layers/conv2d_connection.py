import numpy as np
from math import ceil

from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.base_connection import BaseConnection


conv2d_init = create_custom_init_var_snippet_class(
    'conv2d',

    param_names=[
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
    const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);
    const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);
    const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);
    const int conv_iw = $(conv_iw), conv_ic = $(conv_ic);
    const int conv_ow = $(conv_ow), conv_oc = $(conv_oc);

    const int conv_in_row = ($(id_pre) / conv_ic) / conv_iw;
    const int conv_in_col = ($(id_pre) / conv_ic) % conv_iw;
    const int conv_in_chan = $(id_pre) % conv_ic;

    const int conv_out_row = ($(id_post) / conv_oc) / conv_ow;
    const int conv_out_col = ($(id_post) / conv_oc) % conv_ow;
    const int conv_out_chan = $(id_post) % conv_oc;

    int conv_stride_row = conv_out_row * conv_sh - conv_padh;
    int conv_stride_col = conv_out_col * conv_sw - conv_padw;

    int conv_k_row = conv_in_row - conv_stride_row;
    int conv_k_col = conv_in_col - conv_stride_col;

    if (conv_k_row >= 0 && conv_k_row < conv_kh && conv_k_col >= 0 && conv_k_col < conv_kw) {
        $(value) = $(kernels)[
            conv_k_row * (conv_kw * conv_ic * conv_oc) +
            conv_k_col * (conv_ic * conv_oc) +
            conv_in_chan * (conv_oc) +
            conv_out_chan
        ];
    }
    else {
        $(value) = 0.0;
    }
    ''',
)


class Conv2DConnection(BaseConnection):

    def __init__(self, filters, conv_size, conv_strides=None, conv_padding='valid'):
        super(Conv2DConnection, self).__init__()
        self.filters = filters
        self.conv_size = conv_size
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.conv_padding = PadMode(conv_padding)
        self.conv_output_shape = None


    def compile(self, tg_model):
        super(Conv2DConnection, self).compile(tg_model)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.source.shape
        conv_oh, conv_ow, conv_oc = self.conv_output_shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = (conv_kh - 1) // 2
            conv_padw = (conv_kw - 1) // 2

        weights_init = init_var(conv2d_init, {
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
        super(Conv2DConnection, self).connect(source, target)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = source.shape
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
