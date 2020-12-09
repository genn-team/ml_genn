import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers import SynapseType, PadMode
from tensor_genn.layers.base_synapses import BaseSynapses
from tensor_genn.layers.weight_update_models import signed_static_pulse

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

class AvePool2DDenseSynapses(BaseSynapses):

    def __init__(self, units, pool_size, pool_strides=None, 
                 pool_padding='valid', synapse_type='procedural'):
        super(AvePool2DDenseSynapses, self).__init__()
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)
        self.pool_output_shape = None
        self.synapse_type = SynapseType(synapse_type)

    def compile(self, tg_model):
        super(AvePool2DDenseSynapses, self).compile(tg_model)

        # Procedural initialisation
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

        # Add batch synapse populations
        for batch_i in range(tg_model.batch_size):
            pre_nrn = self.source.nrn[batch_i]
            post_nrn = self.target.nrn[batch_i]
            syn_name = '{}_to_{}_syn_{}'.format(self.source.name, self.target.name, batch_i)

            # Batch master
            if not tg_model.share_weights or batch_i == 0:
                algorithm = ('DENSE_PROCEDURALG' if self.synapse_type == SynapseType.PROCEDURAL 
                             else 'DENSE_INDIVIDUALG')
                model = signed_static_pulse if self.source.signed_spikes else 'StaticPulse'
                
                self.syn[batch_i] = tg_model.g_model.add_synapse_population(
                    syn_name, algorithm, NO_DELAY, pre_nrn, post_nrn,
                    model, {}, {'g': g}, {}, {}, 'DeltaCurr', {}, {})
                self.syn[batch_i].vars['g'].set_extra_global_init_param('weights', self.weights.flatten())

            # Batch slave
            else:
                master_syn_name = '{}_to_{}_syn_0'.format(self.source.name, self.target.name)
                self.syn[batch_i] = tg_model.g_model.add_slave_synapse_population(
                    syn_name, master_syn_name, NO_DELAY, pre_nrn, post_nrn, 'DeltaCurr', {}, {})

    def connect(self, source, target):
        super(AvePool2DDenseSynapses, self).connect(source, target)

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

        output_shape = (self.units, )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(self.pool_output_shape), self.units), dtype=np.float64)
