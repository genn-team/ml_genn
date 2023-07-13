from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .input import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel


class PoissonInput(Neuron, InputBase):
    def __init__(self, signed_spikes=False):
        super(PoissonInput, self).__init__()

        self.signed_spikes = signed_spikes

    def get_model(self, population, dt):
        genn_model = {
            "var_name_types": [("Input", "scalar",
                                VarAccess_READ_ONLY_DUPLICATE)],
            "sim_code":
                """
                const bool spike = $(gennrand_uniform) >= exp(-fabs($(Input)) * DT);
                """,
            "threshold_condition_code":
                """
                $(Input) > 0.0 && spike
                """,
            "is_auto_refractory_required": False}

        # If signed spikes are enabled, add negative threshold condition
        if self.signed_spikes:
            genn_model["negative_threshold_condition_code"] =\
                """
                $(Input_pre) < 0.0 && spike
                """

        return NeuronModel(genn_model, None, {}, {"Input": 0.0})
