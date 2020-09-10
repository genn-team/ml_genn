import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_sparse_connect_init_snippet_class
from pygenn.genn_model import init_connectivity, create_cmlf_class, create_cksf_class
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.StlContainers import UnsignedIntVector
from tensor_genn.layers import ConnectionType, PadMode
from tensor_genn.layers.base_connection import BaseConnection


conv2d_init = create_custom_sparse_connect_init_snippet_class(
    'conv2d',

    param_names=[
        'conv_kh', 'conv_kw',
        'conv_sh', 'conv_sw',
        'conv_padh', 'conv_padw',
        'conv_ih', 'conv_iw', 'conv_ic',
        'conv_oh', 'conv_ow', 'conv_oc',
    ],

    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: (pars[0] // pars[2]) * (pars[1] // pars[3]) * pars[11])(),

    calc_kernel_size_func=create_cksf_class(
        lambda pars: UnsignedIntVector([int(pars[0]), int(pars[1]), int(pars[8]), int(pars[11])]))(),

    row_build_code='''
    const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);
    const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);
    const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);
    const int conv_iw = $(conv_iw), conv_ic = $(conv_ic);
    const int conv_ow = $(conv_ow), conv_oh = $(conv_oh), conv_oc = $(conv_oc);
    
    const int inRow = ($(id_pre) / conv_ic) / conv_iw;
    const int inCol = ($(id_pre) / conv_ic) % conv_iw;
    const int inChan = $(id_pre) % conv_ic;
    const int minOutRow = min(conv_oh, max(0, 1 + ((inRow + conv_padh - conv_kh) / conv_sh)));
    const int maxOutRow = min(conv_oh, max(0, 1 + ((inRow + conv_padh) / conv_sh)));
    const int minOutCol = min(conv_ow, max(0, 1 + ((inCol + conv_padw - conv_kw) / conv_sw)));
    const int maxOutCol = min(conv_ow, max(0, 1 + ((inCol + conv_padw) / conv_sw)));

    for(int outRow = minOutRow; outRow != maxOutRow; outRow++) {
        const int strideRow = (outRow * conv_sh) - conv_padh;
        const int kernRow = inRow - strideRow;
        for(int outCol = minOutCol; outCol < maxOutCol; outCol++) {
            const int strideCol = (outCol * conv_sw) - conv_padw;
            const int kernCol = inCol - strideCol;
            for(int outChan = 0; outChan < conv_oc; outChan++) {
                const int idPost = ((outRow * conv_ow * conv_oc) +
                                    (outCol * conv_oc) +
                                    outChan);
                $(addSynapse, idPost, kernRow, kernCol, inChan, outChan);
            }
        }
    }
    
    // End the row
    // **THINK** beginning to doubt the value of the GeNN-provided outer loop
    $(endRow);
    ''',
)


class Conv2DConnection(BaseConnection):

    def __init__(self, filters, conv_size, conv_strides=None, conv_padding='valid', connection_type='procedural'):
        super(Conv2DConnection, self).__init__()
        self.filters = filters
        self.conv_size = conv_size
        if conv_strides == None:
            self.conv_strides = (1, 1)
        else:
            self.conv_strides = conv_strides
        self.conv_padding = PadMode(conv_padding)
        self.connection_type = ConnectionType(connection_type)


    def compile(self, tg_model):
        super(Conv2DConnection, self).compile(tg_model)

        # Procedural initialisation
        if self.connection_type == ConnectionType.PROCEDURAL:
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

            connect = init_connectivity(conv2d_init, {
                'conv_kh': conv_kh, 'conv_kw': conv_kw,
                'conv_sh': conv_sh, 'conv_sw': conv_sw,
                'conv_padh': conv_padh, 'conv_padw': conv_padw,
                'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
                'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc,
            })

        # Sparse initialisation
        elif self.connection_type == ConnectionType.SPARSE:
            g, indices = self.genn_sparse_weights()

        # Add batch synapse populations
        for batch_i in range(tg_model.batch_size):
            pre_nrn = self.source.nrn[batch_i]
            post_nrn = self.target.nrn[batch_i]
            syn_name = '{}_to_{}_syn_{}'.format(self.source.name, self.target.name, batch_i)

            # Batch master
            if not tg_model.share_weights or batch_i == 0:

                if self.connection_type == ConnectionType.PROCEDURAL:
                    self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                        syn_name, 'PROCEDURAL_KERNELG', NO_DELAY, pre_nrn, post_nrn,
                        'StaticPulse', {}, {'g': self.weights.flatten()}, {}, {}, 'DeltaCurr', {}, {},
                        connect)

                elif self.connection_type == ConnectionType.SPARSE:
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
