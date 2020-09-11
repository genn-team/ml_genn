import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers import ConnectionType, PadMode
from tensor_genn.layers.base_connection import BaseConnection


avepool2d_dense_big_pool_init = create_custom_init_var_snippet_class(
    'avepool2d_dense_big_pool',

    param_names=[
        'pool_kh', 'pool_kw',
        'pool_sh', 'pool_sw',
        'pool_padh', 'pool_padw',
        'pool_ih', 'pool_iw', 'pool_ic',
        'dense_ih', 'dense_iw', 'dense_ic',
        'dense_units',
    ],

    extra_global_params=[
        ('weights', 'scalar*'),
    ],

    var_init_code='''
    const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
    const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
    const int pool_padh = $(pool_padh), pool_padw = $(pool_padw);
    const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);

    const int pool_in_row = ($(id_pre) / pool_ic) / pool_iw;
    const int pool_in_col = ($(id_pre) / pool_ic) % pool_iw;
    const int pool_in_chan = $(id_pre) % pool_ic;

    const int dense_iw = $(dense_iw), dense_ic = $(dense_ic);
    const int dense_units = $(dense_units);

    const int dense_out_unit = $(id_post);

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

            int dense_in_unit = pool_out_row * (dense_iw * dense_ic) + pool_out_col * (dense_ic) + pool_in_chan;

            weight += $(weights)[
                dense_in_unit * (dense_units) +
                dense_out_unit
            ] / (pool_kh_crop * pool_kw_crop);

            pool_out_col--;
            pool_stride_col = pool_out_col * pool_sw - pool_padw;
        }

        pool_out_row--;
        pool_stride_row = pool_out_row * pool_sh - pool_padh;
    }

    $(value) = weight;
    ''',
)

avepool2d_dense_small_pool_init = create_custom_init_var_snippet_class(
    'avepool2d_dense_big_pool',

    param_names=[
        'pool_kh', 'pool_kw',
        'pool_sh', 'pool_sw',
        'pool_padh', 'pool_padw',
        'pool_ih', 'pool_iw', 'pool_ic',
        'dense_ih', 'dense_iw', 'dense_ic',
        'dense_units',
    ],

    extra_global_params=[
        ('weights', 'scalar*'),
    ],

    var_init_code='''
    const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
    const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
    const int pool_padh = $(pool_padh), pool_padw = $(pool_padw);
    const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);

    const int pool_in_row = ($(id_pre) / pool_ic) / pool_iw;
    const int pool_in_col = ($(id_pre) / pool_ic) % pool_iw;
    const int pool_in_chan = $(id_pre) % pool_ic;

    const int dense_iw = $(dense_iw), dense_ic = $(dense_ic);
    const int dense_units = $(dense_units);

    const int dense_out_unit = $(id_post);

    // Calculate corresponding pool output
    const int pool_out_row = (pool_in_row + pool_padh) / pool_sh;
    const int pool_stride_row = pool_out_row * pool_sh - pool_padh;
    const int pool_kh_crop = min(pool_stride_row + pool_kh, pool_ih) - max(pool_stride_row, 0);
    const int pool_out_col = (pool_in_col + pool_padw) / pool_sw;
    const int pool_stride_col = pool_out_col * pool_sw - pool_padw;
    const int pool_kw_crop = min(pool_stride_col + pool_kw, pool_iw) - max(pool_stride_col, 0);

    const int dense_in_unit = pool_out_row * (dense_iw * dense_ic) + pool_out_col * (dense_ic) + pool_in_chan;

    $(value) = $(weights)[
        dense_in_unit * (dense_units) +
        dense_out_unit
    ] / (pool_kh_crop * pool_kw_crop);
    ''',
)

class AvePool2DDenseConnection(BaseConnection):

    def __init__(self, units, pool_size, pool_strides=None, pool_padding='valid', connection_type='procedural'):
        super(AvePool2DDenseConnection, self).__init__()
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)
        self.pool_output_shape = None
        self.connection_type = ConnectionType(connection_type)


    def compile(self, tg_model):
        super(AvePool2DDenseConnection, self).compile(tg_model)

        # Procedural initialisation
        if self.connection_type == ConnectionType.PROCEDURAL:
            pool_kh, pool_kw = self.pool_size
            pool_sh, pool_sw = self.pool_strides
            pool_ih, pool_iw, pool_ic = self.source.shape
            if self.pool_padding == PadMode.VALID:
                pool_padh = 0
                pool_padw = 0
            elif self.pool_padding == PadMode.SAME:
                pool_padh = (pool_kh - 1) // 2
                pool_padw = (pool_kw - 1) // 2

            dense_ih, dense_iw, dense_ic = self.pool_output_shape
            
            # If pool size is greater than stride then a more complex model which 
            # allows pool inputs to appear in multiple pool outputs is required
            model = (avepool2d_dense_big_pool_init if pool_kh > pool_sh or pool_kw > pool_sw 
                     else avepool2d_dense_small_pool_init)
                     
            g = init_var(model, {
                'pool_kh': pool_kh, 'pool_kw': pool_kw,
                'pool_sh': pool_sh, 'pool_sw': pool_sw,
                'pool_padh': pool_padh, 'pool_padw': pool_padw,
                'pool_ih': pool_ih, 'pool_iw': pool_iw, 'pool_ic': pool_ic,
                'dense_ih': dense_ih, 'dense_iw': dense_iw, 'dense_ic': dense_ic,
                'dense_units': self.units,
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
                        syn_name, 'DENSE_PROCEDURALG', NO_DELAY, pre_nrn, post_nrn,
                        'StaticPulse', {}, {'g': g}, {}, {}, 'DeltaCurr', {}, {}
                    )
                    self.syn[batch_i].vars['g'].set_extra_global_init_param('weights', self.weights.flatten())

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
        super(AvePool2DDenseConnection, self).connect(source, target)

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

        self.output_shape = (self.units, )

        if target.shape is None:
            target.shape = self.output_shape
        elif self.output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(self.pool_output_shape), self.units), dtype=np.float64)


    def genn_sparse_weights(self):

        # === AvePool2D Weights ===
        pool_weights = np.zeros((np.prod(self.source.shape), np.prod(self.pool_output_shape)))
        pool_connect = np.zeros(pool_weights.shape, dtype=np.bool)

        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = self.source.shape
        pool_oh, pool_ow, pool_oc = self.pool_output_shape
        if self.pool_padding == PadMode.VALID:
            pool_padh = 0
            pool_padw = 0
        elif self.pool_padding == PadMode.SAME:
            pool_padh = (pool_kh - 1) // 2
            pool_padw = (pool_kw - 1) // 2

        # For each in {one-to-one input -> output channel}:
        for channel in range(pool_ic):
            pool_chan_weights = pool_weights[channel::pool_ic, channel::pool_oc]
            pool_chan_connect = pool_connect[channel::pool_ic, channel::pool_oc]

            # For each AvePool2D output pixel:
            for pool_out_row in range(pool_oh):
                pool_stride_row = pool_out_row * pool_sh - pool_padh
                for pool_out_col in range(pool_ow):
                    pool_stride_col = pool_out_col * pool_sw - pool_padw

                    # Get a weights view for this out pixel.
                    pool_out_pixel_weights = pool_chan_weights[:, pool_out_row * pool_ow + pool_out_col]
                    pool_out_pixel_weights.shape = (pool_ih, pool_iw)
                    pool_out_pixel_connect = pool_chan_connect[:, pool_out_row * pool_ow + pool_out_col]
                    pool_out_pixel_connect.shape = (pool_ih, pool_iw)

                    # Get a weights view for this cropped stride.
                    crop_T = max(pool_stride_row, 0)
                    crop_B = min(pool_stride_row + pool_kh, pool_ih)
                    crop_L = max(pool_stride_col, 0)
                    crop_R = min(pool_stride_col + pool_kw, pool_iw)
                    pool_stride_weights = pool_out_pixel_weights[crop_T:crop_B, crop_L:crop_R]
                    pool_stride_connect = pool_out_pixel_connect[crop_T:crop_B, crop_L:crop_R]

                    # Set weights for this stride.
                    pool_stride_weights[:] = 1.0 / pool_stride_weights.size
                    pool_stride_connect[:] = True

        # === Dense Weights ===
        dense_weights = self.weights
        dense_connect = np.ones(dense_weights.shape, dtype=np.bool)

        # === Combined Weights ===
        combined_weights = np.zeros((pool_weights.shape[0], dense_weights.shape[1]))
        combined_connect = np.zeros(combined_weights.shape, dtype=np.bool)

        # For each in {one-to-one input -> output channel}:
        for channel in range(pool_ic):
            pool_chan_weights = pool_weights[channel::pool_ic, channel::pool_ic]
            pool_chan_connect = pool_connect[channel::pool_ic, channel::pool_ic]
            dense_chan_weights = dense_weights[channel::pool_ic, :]
            dense_chan_connect = dense_connect[channel::pool_ic, :]
            combined_chan_weights = combined_weights[channel::pool_ic, :]
            combined_chan_connect = combined_connect[channel::pool_ic, :]

            # Set weights to dot product of AvePool2D and Dense weights.
            combined_chan_weights[:] = np.dot(pool_chan_weights, dense_chan_weights)
            combined_chan_connect[:] = np.dot(pool_chan_connect, dense_chan_connect)

        # === Weight Values and Indices ===
        w_indices = np.nonzero(combined_connect)
        w_values = combined_weights[w_indices]
        return w_values, w_indices
