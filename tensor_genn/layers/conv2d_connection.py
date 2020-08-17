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

    group_params=[
        ('conv_kh_reg', 'int', '$(conv_kh)'),
        ('conv_kw_reg', 'int', '$(conv_kw)'),
        ('conv_sh_reg', 'int', '$(conv_sh)'),
        ('conv_sw_reg', 'int', '$(conv_sw)'),
        ('conv_padh_reg', 'int', '$(conv_padh)'),
        ('conv_padw_reg', 'int', '$(conv_padw)'),
        ('conv_iw_reg', 'int', '$(conv_iw)'),
        ('conv_ic_reg', 'int', '$(conv_ic)'),
        ('conv_ow_reg', 'int', '$(conv_ow)'),
        ('conv_oc_reg', 'int', '$(conv_oc)')
    ],

    pre_params=[
        ('conv_in_row', 'int', '($(id_pre) / $(conv_ic_reg)) / $(conv_iw_reg)'),
        ('conv_in_col', 'int', '($(id_pre) / $(conv_ic_reg)) % $(conv_iw_reg)'),
        ('conv_in_chan', 'int', '$(id_pre) % $(conv_ic_reg)')
    ],

    post_params=[
        ('conv_out_row', 'int', '($(id_post) / $(conv_oc_reg)) / $(conv_ow_reg)'),
        ('conv_out_col', 'int', '($(id_post) / $(conv_oc_reg)) % $(conv_ow_reg)'),
        ('conv_out_chan', 'int', '$(id_post) % $(conv_oc_reg)')
    ],

    extra_global_params=[
        ('kernels', 'scalar*'),
    ],

    var_init_code='''
    const int conv_stride_row = $(conv_out_row) * $(conv_sh_reg) - $(conv_padh_reg);
    const int conv_stride_col = $(conv_out_col) * $(conv_sw_reg) - $(conv_padw_reg);
    const int conv_k_row = $(conv_in_row) - conv_stride_row;
    const int conv_k_col = $(conv_in_col) - conv_stride_col;
    if (conv_k_row >= 0 && conv_k_row < $(conv_kh_reg) && conv_k_col >= 0 && conv_k_col < $(conv_kw_reg)) {
        $(value) = $(kernels)[
            conv_k_row * ($(conv_kw_reg) * $(conv_ic_reg) * $(conv_oc_reg)) +
            conv_k_col * ($(conv_ic_reg) * $(conv_oc_reg)) +
            $(conv_in_chan) * $(conv_oc_reg) +
            $(conv_out_chan)
        ];
    }
    else {
        $(value) = 0.0;
    }
    ''',
)


class Conv2DConnection(BaseConnection):

    def __init__(self, filters, conv_size, conv_strides=None, conv_padding='valid', genn_procedural=True):
        super(Conv2DConnection, self).__init__()
        self.filters = filters
        self.conv_size = conv_size
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.conv_padding = PadMode(conv_padding)
        self.genn_procedural = genn_procedural


    def compile(self, tg_model):
        super(Conv2DConnection, self).compile(tg_model)

        # Procedural initialisation
        if self.genn_procedural:
            conv_kh, conv_kw = self.conv_size
            conv_sh, conv_sw = self.conv_strides
            conv_ih, conv_iw, conv_ic = self.source.shape
            conv_oh, conv_ow, conv_oc = self.target.shape
            if self.conv_padding == PadMode.VALID:
                conv_padh = 0
                conv_padw = 0
            elif self.conv_padding == PadMode.SAME:
                conv_padh = (conv_kh - 1) // 2
                conv_padw = (conv_kw - 1) // 2

            g = init_var(conv2d_init, {
                'conv_kh': conv_kh, 'conv_kw': conv_kw,
                'conv_sh': conv_sh, 'conv_sw': conv_sw,
                'conv_padh': conv_padh, 'conv_padw': conv_padw,
                'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
                'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc,
            })

        # Sparse initialisation
        else:
            g, indices = self.genn_sparse_weights()

        # Add batch synapse populations
        for batch_i in range(tg_model.batch_size):
            pre_nrn = self.source.nrn[batch_i]
            post_nrn = self.target.nrn[batch_i]
            syn_name = '{}_to_{}_syn_{}'.format(self.source.name, self.target.name, batch_i)

            # Batch master
            if not tg_model.share_weights or batch_i == 0:

                if self.genn_procedural:
                    self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                        syn_name, 'DENSE_PROCEDURALG', NO_DELAY, pre_nrn, post_nrn,
                        'StaticPulse', {}, {'g': g}, {}, {}, 'DeltaCurr', {}, {}
                    )
                    self.syn[batch_i].vars['g'].set_extra_global_init_param('kernels', self.weights.flatten())

                else:
                    self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                        syn_name, 'SPARSE_INDIVIDUALG', NO_DELAY, pre_nrn, post_nrn,
                        'StaticPulse', {}, {'g': g}, {}, {}, 'DeltaCurr', {}, {}
                    )
                    self.syn[batch_i].set_sparse_connections(indices[0], indices[1])

            # Batch slave
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
            self.output_shape = (
                ceil(float(conv_ih - conv_kh + 1) / float(conv_sh)),
                ceil(float(conv_iw - conv_kw + 1) / float(conv_sw)),
                self.filters,
            )
        elif self.conv_padding == PadMode.SAME:
            self.output_shape = (
                ceil(float(conv_ih) / float(conv_sh)),
                ceil(float(conv_iw) / float(conv_sw)),
                self.filters,
            )

        if target.shape is None:
            target.shape = self.output_shape
        elif self.output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((conv_kh, conv_kw, conv_ic, self.filters), dtype=np.float64)


    def genn_sparse_weights(self):

        # === Conv2D Weights ===
        conv_weights = np.zeros((np.prod(self.source.shape), np.prod(self.target.shape)))
        conv_connect = np.zeros(conv_weights.shape, dtype=np.bool)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.source.shape
        conv_oh, conv_ow, conv_oc = self.target.shape
        if self.conv_padding == PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding == PadMode.SAME:
            conv_padh = (conv_kh - 1) // 2
            conv_padw = (conv_kw - 1) // 2

        # For each in {all-to-all input -> output channel}:
        for in_channel in range(conv_ic):
            for out_channel in range(conv_oc):
                conv_chan_kernel = self.weights[:, :, in_channel, out_channel]
                conv_chan_weights = conv_weights[in_channel::conv_ic, out_channel::conv_oc]
                conv_chan_connect = conv_connect[in_channel::conv_ic, out_channel::conv_oc]

                # For each Conv2D output pixel: 
                for conv_out_row in range(conv_oh):
                    conv_stride_row = conv_out_row * conv_sh - conv_padh
                    for conv_out_col in range(conv_ow):
                        conv_stride_col = conv_out_col * conv_sw - conv_padw

                        # Get a weights view for this out pixel.
                        conv_out_pixel_weights = conv_chan_weights[:, conv_out_row * conv_ow + conv_out_col]
                        conv_out_pixel_weights.shape = (conv_ih, conv_iw)
                        conv_out_pixel_connect = conv_chan_connect[:, conv_out_row * conv_ow + conv_out_col]
                        conv_out_pixel_connect.shape = (conv_ih, conv_iw)

                        # Get a weights view for this cropped stride.
                        crop_T = max(conv_stride_row, 0)
                        crop_B = min(conv_stride_row + conv_kh, conv_ih)
                        crop_L = max(conv_stride_col, 0)
                        crop_R = min(conv_stride_col + conv_kw, conv_iw)
                        conv_stride_weights = conv_out_pixel_weights[crop_T:crop_B, crop_L:crop_R]
                        conv_stride_connect = conv_out_pixel_connect[crop_T:crop_B, crop_L:crop_R]

                        # Get a cropped kernel view.
                        crop_T =       0 - min(conv_stride_row, 0)
                        crop_B = conv_kh - max(conv_stride_row + conv_kh - conv_ih, 0)
                        crop_L =       0 - min(conv_stride_col, 0)
                        crop_R = conv_kw - max(conv_stride_col + conv_kw - conv_iw, 0)
                        conv_cropped_kernel = conv_chan_kernel[crop_T:crop_B, crop_L:crop_R]

                        # Set weights for this stride.
                        conv_stride_weights[:] = conv_cropped_kernel
                        conv_stride_connect[:] = True

        # === Weight Values and Indices ===
        w_indices = np.nonzero(conv_connect)
        w_values = conv_weights[w_indices]
        return w_values, w_indices
