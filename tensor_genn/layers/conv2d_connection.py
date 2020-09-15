import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_sparse_connect_init_snippet_class
from pygenn.genn_model import init_connectivity, create_cmlf_class, create_cksf_class
from pygenn.genn_wrapper import NO_DELAY, init_var_kernel
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
    // Stash all parameters in registers
    // **NOTE** this means parameters from group structure only get converted from float->int once
    // **NOTE** if they're actually constant, compiler is still likely to treat them as constants rather than allocating registers
    const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);
    const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);
    const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);
    const int conv_iw = $(conv_iw), conv_ic = $(conv_ic);
    const int conv_ow = $(conv_ow), conv_oh = $(conv_oh), conv_oc = $(conv_oc);

    // Convert presynaptic neuron ID to row, column and channel
    const int inRow = ($(id_pre) / conv_ic) / conv_iw;
    const int inCol = ($(id_pre) / conv_ic) % conv_iw;
    const int inChan = $(id_pre) % conv_ic;

    // Calculate range of output rows and columns which this presynaptic neuron will connect to
    const int minOutRow = min(conv_oh, max(0, 1 + ((inRow + conv_padh - conv_kh) / conv_sh)));
    const int maxOutRow = min(conv_oh, max(0, 1 + ((inRow + conv_padh) / conv_sh)));
    const int minOutCol = min(conv_ow, max(0, 1 + ((inCol + conv_padw - conv_kw) / conv_sw)));
    const int maxOutCol = min(conv_ow, max(0, 1 + ((inCol + conv_padw) / conv_sw)));

    // Loop through output rows, columns and channels
    for(int outRow = minOutRow; outRow != maxOutRow; outRow++) {
        const int strideRow = (outRow * conv_sh) - conv_padh;
        const int kernRow = inRow - strideRow;
        for(int outCol = minOutCol; outCol < maxOutCol; outCol++) {
            const int strideCol = (outCol * conv_sw) - conv_padw;
            const int kernCol = inCol - strideCol;
            for(int outChan = 0; outChan < conv_oc; outChan++) {
                // Calculate postsynaptic index and add synapse
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

        connectivity_init = init_connectivity(conv2d_init, {
            'conv_kh': conv_kh, 'conv_kw': conv_kw,
            'conv_sh': conv_sh, 'conv_sw': conv_sw,
            'conv_padh': conv_padh, 'conv_padw': conv_padw,
            'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
            'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc})

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
                        connectivity_init)

                elif self.connection_type == ConnectionType.SPARSE:
                    self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                        syn_name, 'SPARSE_INDIVIDUALG', NO_DELAY, pre_nrn, post_nrn,
                        'StaticPulse', {}, {'g': init_var_kernel()}, {}, {}, 'DeltaCurr', {}, {},
                        connectivity_init)
                    self.syn[batch_i].vars['g'].set_extra_global_init_param('kernel', self.weights.flatten())

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