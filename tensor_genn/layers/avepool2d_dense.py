import numpy as np
from math import ceil

from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY

from tensor_genn.layers.base_connection import PadMode
from tensor_genn.layers.base_connection import BaseConnection
from tensor_genn.layers.layer import Layer
from tensor_genn.layers.neuron_models import if_model


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


# # TEMP DEBUG
# avepool2d_dense_init = create_custom_init_var_snippet_class(
#     'avepool2d_dense',

#     param_names=[
#         'dense_units',
#     ],

#     extra_global_params=[
#         ('weights', 'scalar*'),
#     ],

#     var_init_code='''
#     $(value) = 0.0;
#     '''
# )


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

        # # TEMP DEBUG
        # weights_init = init_var(avepool2d_dense_init, {
        #     'dense_units': self.units,
        # })

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


class AvePool2DDense(Layer):

    def __init__(self, name, model, params, vars_init, global_params,
                 units, pool_size, pool_strides=None, pool_padding='valid'):
        super(AvePool2DDense, self).__init__(name, model, params, vars_init, global_params)
        self.units = units
        self.pool_size = pool_size
        if pool_strides == None:
            self.pool_strides = (pool_size[0], pool_size[1])
        else:
            self.pool_strides = pool_strides
        self.pool_padding = PadMode(pool_padding)


    def connect(self, sources):
        connections = [
            AvePool2DDenseConnection(self.units, self.pool_size, self.pool_strides, self.pool_padding)
            for i in range(len(sources))
        ]
        super(AvePool2DDense, self).connect(sources, connections)


class IFAvePool2DDense(AvePool2DDense):

    def __init__(self, name, units, pool_size, pool_strides=None, pool_padding='valid', threshold=1.0):
        super(IFAvePool2DDense, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            units, pool_size, pool_strides, pool_padding
        )

        self.threshold = threshold


    def set_threshold(self, threshold):
        if not self.tg_model:
            raise RuntimeError('model must be compiled before calling set_threshold')

        for batch_i in range(self.tg_model.batch_size):
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            nrn = self.tg_model.g_model.neuron_populations[nrn_name]
            nrn.extra_global_params['Vthr'].view[:] = threshold

        self.threshold = threshold
