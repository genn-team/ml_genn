import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var

from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers.base_synapses import BaseSynapses
from ml_genn.layers.weight_update_models import signed_static_pulse
from ml_genn.layers.helper import _get_param_2d

avepool2d_init = create_custom_init_var_snippet_class(
    'avepool2d',

    param_names=[
        'pool_kh', 'pool_kw',
        'pool_sh', 'pool_sw',
        'pool_padh', 'pool_padw',
        'pool_ih', 'pool_iw', 'pool_ic',
        'pool_oh', 'pool_ow', 'pool_oc',
    ],

    var_init_code='''
    const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
    const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
    const int pool_padh = $(pool_padh), pool_padw = $(pool_padw);
    const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);

    // Convert presynaptic neuron ID to row, column and channel in pool input
    const int poolInRow = ($(id_pre) / pool_ic) / pool_iw;
    const int poolInCol = ($(id_pre) / pool_ic) % pool_iw;
    const int poolInChan = $(id_pre) % pool_ic;

    // Calculate corresponding pool output
    const int poolOutRow = (poolInRow + pool_padh) / pool_sh;
    const int poolStrideRow = poolOutRow * pool_sh - pool_padh;
    const int poolCropKH = min(poolStrideRow + pool_kh, pool_ih) - max(poolStrideRow, 0);
    const int poolOutCol = (poolInCol + pool_padw) / pool_sw;
    const int poolStrideCol = poolOutCol * pool_sw - pool_padw;
    const int poolCropKW = min(poolStrideCol + pool_kw, pool_iw) - max(poolStrideCol, 0);

    $(value) = 0.0;
    if ((poolInRow < (poolStrideRow + pool_kh)) && (poolInCol < (poolStrideCol + pool_kw))) {

        const int pool_ow = $(pool_ow), pool_oc = $(pool_oc);

        const int poolOutId = poolOutRow * (pool_ow * pool_oc) + poolOutCol * (pool_oc) + poolInChan;

        if (poolOutId == $(id_post)) {
            $(value) = 1.0 / (poolCropKH * poolCropKW);
        }
    }
    ''',
)

class AvePool2DSynapses(BaseSynapses):

    def __init__(self, pool_size, pool_strides=None,
                 pool_padding='valid', connectivity_type='procedural'):
        super(AvePool2DSynapses, self).__init__()
        self.pool_size = _get_param_2d('pool_size', pool_size)
        self.pool_strides = _get_param_2d('pool_strides', pool_strides, default=self.pool_size)
        self.pool_padding = PadMode(pool_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)
        if self.pool_strides[0] < self.pool_size[0] or self.pool_strides[1] < self.pool_size[1]:
            raise NotImplementedError('pool stride < pool size is not supported')

    def connect(self, source, target):
        super(AvePool2DSynapses, self).connect(source, target)

        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        if self.pool_padding == PadMode.VALID:
            output_shape = (
                ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
                ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
                pool_ic,
            )
        elif self.pool_padding == PadMode.SAME:
            output_shape = (
                ceil(float(pool_ih) / float(pool_sh)),
                ceil(float(pool_iw) / float(pool_sw)),
                pool_ic,
            )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty(0, dtype=np.float64)

    def compile(self, mlg_model, name):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = self.source().shape
        pool_oh, pool_ow, pool_oc = self.target().shape
        if self.pool_padding == PadMode.VALID:
            pool_padh = 0
            pool_padw = 0
        elif self.pool_padding == PadMode.SAME:
            pool_padh = (pool_kh - 1) // 2
            pool_padw = (pool_kw - 1) // 2

        wu_var_init = init_var(avepool2d_init, {
            'pool_kh': pool_kh, 'pool_kw': pool_kw,
            'pool_sh': pool_sh, 'pool_sw': pool_sw,
            'pool_padh': pool_padh, 'pool_padw': pool_padw,
            'pool_ih': pool_ih, 'pool_iw': pool_iw, 'pool_ic': pool_ic,
            'pool_oh': pool_oh, 'pool_ow': pool_ow, 'pool_oc': pool_oc})

        conn = ('DENSE_PROCEDURALG' if self.connectivity_type == ConnectivityType.PROCEDURAL
                else 'DENSE_INDIVIDUALG')
        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': wu_var_init}

        super(AvePool2DSynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                               {}, {}, 'DeltaCurr', {}, {}, None, {})
