import logging
import numpy as np

from pygenn.genn_model import create_custom_sparse_connect_init_snippet_class
from pygenn.genn_model import create_cmlf_class
from pygenn.genn_model import create_cksf_class
from pygenn.genn_model import init_connectivity
from pygenn.genn_model import init_var
from pygenn.genn_wrapper.StlContainers import UnsignedIntVector

from ml_genn.layers import ConnectivityType, PadMode
from ml_genn.layers.base_synapses import BaseSynapses
from ml_genn.layers.weight_update_models import signed_static_pulse
from ml_genn.layers.helper import _get_conv_same_padding, _get_param_2d

logger = logging.getLogger(__name__)

convtranspose2d_init = create_custom_sparse_connect_init_snippet_class(
    'convtranspose2d',

    param_names=[
        'conv_kh', 'conv_kw',
        'conv_sh', 'conv_sw',
        'conv_padh', 'conv_padw',
        'conv_ih', 'conv_iw', 'conv_ic',
        'conv_oh', 'conv_ow', 'conv_oc',
    ],

    calc_max_row_len_func=create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[0] * pars[1] * pars[11]))(),

    calc_kernel_size_func=create_cksf_class(
        lambda pars: UnsignedIntVector([int(pars[0]), int(pars[1]), int(pars[11]), int(pars[8])]))(),

    row_build_state_vars=[
        ('inRow', 'int', '($(id_pre) / (int) $(conv_ic)) / (int) $(conv_iw)'),
        ('inCol', 'int', '($(id_pre) / (int) $(conv_ic)) % (int) $(conv_iw)'),
        ('inChan', 'int', '$(id_pre) % (int) $(conv_ic)'),
        ('strideRow', 'int', '(inRow * (int) $(conv_sh)) - (int) $(conv_padh)'),
        ('strideCol', 'int', '(inCol * (int) $(conv_sw)) - (int) $(conv_padw)'),
        ('outRow', 'int', 'min((int) $(conv_oh), max(0, (inRow * (int) $(conv_sh)) - (int) $(conv_padh)))'),
        ('maxOutRow', 'int', 'min((int) $(conv_oh), max(0, (inRow * (int) $(conv_sh)) + (int) $(conv_kh) - (int) $(conv_padh)))'),
        ('minOutCol', 'int', 'min((int) $(conv_ow), max(0, (inCol * (int) $(conv_sw)) - (int) $(conv_padw)))'),
        ('maxOutCol', 'int', 'min((int) $(conv_ow), max(0, (inCol * (int) $(conv_sw)) + (int) $(conv_kw) - (int) $(conv_padw)))'),
    ],

    row_build_code='''
    if ($(outRow) == $(maxOutRow)) {
       $(endRow);
    }
    const int kernRow = $(outRow) - $(strideRow);
    for (int outCol = $(minOutCol); outCol < $(maxOutCol); outCol++) {
        const int kernCol = outCol - $(strideCol);
        for (unsigned int outChan = 0; outChan < (unsigned int) $(conv_oc); outChan++) {
            const int idPost = (($(outRow) * (int) $(conv_ow) * (int) $(conv_oc)) +
                                (outCol * (int) $(conv_oc)) +
                                outChan);
            $(addSynapse, idPost, kernRow, kernCol, outChan, inChan);
        }
    }
    $(outRow)++;
    ''',
)

class ConvTranspose2DSynapses(BaseSynapses):

    def __init__(self, filters, conv_size, conv_strides=None,
                 conv_padding='valid', connectivity_type='procedural'):
        super(ConvTranspose2DSynapses, self).__init__()
        self.filters = filters
        self.conv_size = _get_param_2d('conv_size', conv_size)
        self.conv_strides = _get_param_2d('conv_strides', conv_strides, default=(1, 1))
        self.conv_padding = PadMode(conv_padding)
        self.connectivity_type = ConnectivityType(connectivity_type)

    def connect(self, source, target):
        super(ConvTranspose2DSynapses, self).connect(source, target)

        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = source.shape
        if self.conv_padding is PadMode.VALID:
            output_shape = (
                conv_ih * conv_sh + max(conv_kh - conv_sh, 0),
                conv_iw * conv_sw + max(conv_kw - conv_sw, 0),
                self.filters,
            )
        elif self.conv_padding is PadMode.SAME:
            output_shape = (
                conv_ih * conv_sh,
                conv_iw * conv_sw,
                self.filters,
            )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((conv_kh, conv_kw, self.filters, conv_ic), dtype=np.float64)

    def compile(self, mlg_model, name):
        conv_kh, conv_kw = self.conv_size
        conv_sh, conv_sw = self.conv_strides
        conv_ih, conv_iw, conv_ic = self.source().shape
        conv_oh, conv_ow, conv_oc = self.target().shape
        if self.conv_padding is PadMode.VALID:
            conv_padh = 0
            conv_padw = 0
        elif self.conv_padding is PadMode.SAME:
            conv_padh = _get_conv_same_padding(conv_ih, conv_kh, conv_sh)
            conv_padw = _get_conv_same_padding(conv_iw, conv_kw, conv_sw)

        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'

        conn_init = init_connectivity(convtranspose2d_init, {
            'conv_kh': conv_kh, 'conv_kw': conv_kw,
            'conv_sh': conv_sh, 'conv_sw': conv_sw,
            'conv_padh': conv_padh, 'conv_padw': conv_padw,
            'conv_ih': conv_ih, 'conv_iw': conv_iw, 'conv_ic': conv_ic,
            'conv_oh': conv_oh, 'conv_ow': conv_ow, 'conv_oc': conv_oc})

        if self.connectivity_type is ConnectivityType.SPARSE:
            conn = 'SPARSE_INDIVIDUALG'
            wu_var = {'g': init_var('Kernel', {})}
            wu_var_egp = {'g': {'kernel': self.weights.flatten()}}
        elif self.connectivity_type is ConnectivityType.PROCEDURAL:
            conn = 'PROCEDURAL_KERNELG'
            wu_var = {'g': self.weights.flatten()}
            wu_var_egp = {}
        elif self.connectivity_type is ConnectivityType.TOEPLITZ:
            logger.warning("Falling back to procedural connectivity "
                           "for ConvTranspose2DSynapses")
            conn = "PROCEDURAL_KERNELG"
            wu_var = {'g': self.weights.flatten()}
            wu_var_egp = {}

        super(ConvTranspose2DSynapses, self).compile(
            mlg_model, name, conn, wu_model, {}, wu_var, {}, {}, 'DeltaCurr', {}, {},
            conn_init, wu_var_egp)
