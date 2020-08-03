import numpy as np
from math import ceil

from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.base_connection import BaseConnection


avepool2d_dense_init = create_custom_init_var_snippet_class(
    'avepool2d_dense',

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
    
    group_params=[
        ('pool_kh_reg', 'int', '$(pool_kh)'), 
        ('pool_kw_reg', 'int', '$(pool_kw)'),
        ('pool_sh_reg', 'int', '$(pool_sh)'), 
        ('pool_sw_reg', 'int', '$(pool_sw)'),
        ('pool_padh_reg', 'int', '$(pool_padh)'), 
        ('pool_padw_reg', 'int', '$(pool_padw)'),
        ('pool_ih_reg', 'int', '$(pool_ih)'), 
        ('pool_iw_reg', 'int', '$(pool_iw)'), 
        ('pool_ic_reg', 'int', '$(pool_ic)')],
    
    pre_params = [
        ('pool_in_row', 'int', '($(id_pre) / $(pool_ic_reg)) / $(pool_iw_reg)'),
        ('pool_in_col', 'int', '($(id_pre) / $(pool_ic_reg)) % $(pool_iw_reg)'),
        ('pool_in_chan', 'int', '$(id_pre) % $(pool_ic_reg)')],

  
    var_init_code='''
    const int dense_iw = $(dense_iw), dense_ic = $(dense_ic);
    const int dense_units = $(dense_units);

    const int dense_out_unit = $(id_post);

    scalar weight = 0.0;

    // process only strides with rows containing pool_in_row
    int pool_out_row = ($(pool_in_row) + $(pool_padh_reg)) / $(pool_sh_reg);
    int pool_stride_row = pool_out_row * $(pool_sh_reg) - $(pool_padh_reg);
    while ((pool_stride_row >= -$(pool_padh_reg)) && (pool_stride_row + $(pool_kh_reg) > $(pool_in_row))) {

        int pool_kh_crop = min(pool_stride_row + $(pool_kh_reg), $(pool_ih_reg)) - max(pool_stride_row, 0);

        // process only strides with cols containing pool_in_col
        int pool_out_col = ($(pool_in_col) + $(pool_padw_reg)) / $(pool_sw_reg);
        int pool_stride_col = pool_out_col * $(pool_sw_reg) - $(pool_padw_reg);
        while ((pool_stride_col >= -$(pool_padw_reg)) && (pool_stride_col + $(pool_kw_reg) > $(pool_in_col))) {

            const int pool_kw_crop = min(pool_stride_col + $(pool_kw_reg), $(pool_iw_reg)) - max(pool_stride_col, 0);

            const int dense_in_unit = pool_out_row * (dense_iw * dense_ic) + pool_out_col * (dense_ic) + $(pool_in_chan);

            weight += $(weights)[
                dense_in_unit * (dense_units) +
                dense_out_unit
            ] / (pool_kh_crop * pool_kw_crop);

            pool_out_col--;
            pool_stride_col = pool_out_col * $(pool_sw_reg) - $(pool_padw_reg);
        }

        pool_out_row--;
        pool_stride_row = pool_out_row * $(pool_sh_reg) - $(pool_padh_reg);
    }

    $(value) = weight;
    ''',
)


class AvePool2DDenseConnection(BaseConnection):

    def __init__(self, units, pool_size, pool_strides=None, pool_padding='valid'):
        super(AvePool2DDenseConnection, self).__init__()
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)
        self.pool_output_shape = None
        self.dense_output_shape = None


    def compile(self, tg_model):
        super(AvePool2DDenseConnection, self).compile(tg_model)

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

        weights_init = init_var(avepool2d_dense_init, {
            'pool_kh': pool_kh, 'pool_kw': pool_kw,
            'pool_sh': pool_sh, 'pool_sw': pool_sw,
            'pool_padh': pool_padh, 'pool_padw': pool_padw,
            'pool_ih': pool_ih, 'pool_iw': pool_iw, 'pool_ic': pool_ic,
            'dense_ih': dense_ih, 'dense_iw': dense_iw, 'dense_ic': dense_ic,
            'dense_units': self.units,
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
                self.syn[batch_i].vars['g'].set_extra_global_init_param('weights', self.weights.flatten())

            # Batch slave synapses
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

        self.dense_output_shape = (self.units, )

        if target.shape is None:
            target.shape = self.dense_output_shape
        elif self.dense_output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(self.pool_output_shape), self.units), dtype=np.float64)
