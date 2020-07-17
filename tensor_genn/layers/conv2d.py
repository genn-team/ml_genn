from math import ceil
from enum import Enum
import numpy as np
from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import create_dpf_class
from pygenn.genn_model import init_var
from pygenn.genn_wrapper import NO_DELAY
from tensor_genn.genn_models import if_model


class Conv2DPadMode(Enum):
    VALID = 'valid'
    SAME = 'same'


# === Conv2D initialise class ===
conv2d_init = create_custom_init_var_snippet_class(
    'conv2d',

    param_names=[
        'kh', 'kw',
        'sh', 'sw',
        'ih', 'iw', 'ic',
        'oh', 'ow', 'oc',
        'padh', 'padw',
    ],

    extra_global_params=[
        ('kernels', 'scalar*'),
    ],

    var_init_code='''
    const int kh = $(kh), kw = $(kw);
    const int sh = $(sh), sw = $(sw);
    const int iw = $(iw), ic = $(ic);
    const int ow = $(ow), oc = $(oc);

    int in_row = ($(id_pre) / ic) / iw;
    int in_col = ($(id_pre) / ic) % iw;
    int in_chan = $(id_pre) % ic;

    int out_row = ($(id_post) / oc) / ow;
    int out_col = ($(id_post) / oc) % ow;
    int out_chan = $(id_post) % oc;

    int k_offset_row = out_row * sh - $(padh);
    int k_offset_col = out_col * sw - $(padw);

    int k_row = in_row - k_offset_row;
    int k_col = in_col - k_offset_col;

    if (k_row >= 0 && k_row < kh && k_col >= 0 && k_col < kw) {
        $(value) = $(kernels)[k_row * (kw * ic * oc) + k_col * (ic * oc) + in_chan * (oc) + out_chan];
    } else {
        $(value) = 0.0;
    }
    ''',
)


class Conv2D(object):

    def __init__(self, model, params, vars_init, global_params,
                 name, filters, kernel_size, strides=(1, 1), padding='valid'):
        self.name = name
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = Conv2DPadMode(padding)
        self.model = model
        self.params = params
        self.vars_init = vars_init
        self.global_params = global_params

        self.downstream_layers = []
        self.upstream_layer = None
        self.weights = None
        self.shape = None

        self.tg_model = None


    def connect(self, upstream_layer):
        upstream_layer.downstream_layers.append(self)
        self.upstream_layer = upstream_layer

        kh, kw = self.kernel_size
        sh, sw = self.strides
        ih, iw, ic = self.upstream_layer.shape

        self.weights = np.empty((kh, kw, ic, self.filters), dtype=np.float64)

        if self.padding == Conv2DPadMode.VALID:
            self.shape = (
                ceil(float(ih - kh + 1) / float(sh)),
                ceil(float(iw - kw + 1) / float(sw)),
                self.filters,
            )

        elif self.padding == Conv2DPadMode.SAME:
            self.shape = (
                ceil(float(ih) / float(sh)),
                ceil(float(iw) / float(sw)),
                self.filters,
            )


    def set_weights(self, weights):
        self.weights[:] = weights


    def get_weights(self):
        return self.weights.copy()


    def compile(self, tg_model):
        self.tg_model = tg_model

        kh, kw = self.kernel_size
        sh, sw = self.strides
        ih, iw, ic = self.upstream_layer.shape
        oh, ow, oc = self.shape

        if self.padding == Conv2DPadMode.VALID:
            padh = 0
            padw = 0

        elif self.padding == Conv2DPadMode.SAME:
            padh = (kh - 1) // 2
            padw = (kw - 1) // 2

        weights_init = init_var(conv2d_init, {
            'kh': kh, 'kw': kw,
            'sh': sh, 'sw': sw,
            'ih': ih, 'iw': iw, 'ic': ic,
            'oh': oh, 'ow': ow, 'oc': oc,
            'padh': padh, 'padw': padw,
        })

        post_nrn_n = np.prod(self.shape)
        for batch_i in range(tg_model.batch_size):

            # Add neuron population
            post_nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            post_nrn = tg_model.g_model.add_neuron_population(
                post_nrn_name, post_nrn_n, self.model, self.params, self.vars_init
            )
            for gp in self.global_params:
                post_nrn.set_extra_global_param(gp, self.global_params[gp])

            pre_nrn_name = '{}_nrn_{}'.format(self.upstream_layer.name, batch_i)
            syn_name = '{}_to_{}_syn_{}'.format(self.upstream_layer.name, self.name, batch_i)

            # Batch master synapses
            if not tg_model.share_weights or batch_i == 0:
                syn = tg_model.g_model.add_synapse_population(
                    syn_name, 'DENSE_PROCEDURALG', NO_DELAY, pre_nrn_name, post_nrn_name,
                    'StaticPulse', {}, {'g': weights_init}, {}, {}, 'DeltaCurr', {}, {}
                )
                syn.vars['g'].set_extra_global_init_param('kernels', self.weights.flatten())

            # Batch slave synapses
            else:
                master_syn_name = '{}_to_{}_syn_0'.format(self.upstream_layer.name, self.name)
                syn = tg_model.g_model.add_slave_synapse_population(
                    syn_name, master_syn_name, NO_DELAY, pre_nrn_name, post_nrn_name, 'DeltaCurr', {}, {}
                )


class IFConv2D(Conv2D):

    def __init__(self, name, filters, kernel_size, strides=(1, 1), padding='valid', threshold=1.0):
        super(IFConv2D, self).__init__(
            if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold},
            name, filters, kernel_size, strides, padding
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
