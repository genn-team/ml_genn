from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .input import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel


class BinarySpikeInput(Neuron, InputBase):
    def __init__(self, signed_spikes=False):
        super(BinarySpikeInput, self).__init__()

        self.signed_spikes = signed_spikes

    def get_model(self, population, batch_size):
        genn_model = {
            "sim_code":
                """
                const bool spike = $(Input) != 0.0;
                """,
            "threshold_condition_code":
                """
                $(Isyn) > 0.0 && spike
                """,
            "is_auto_refractory_required": False}

        # If signed spikes are enabled, add negative threshold condition
        if self.signed_spikes:
            genn_model["negative_threshold_condition_code"] =\
                """
                $(Input) < 0.0 && spike
                """

        neuron_model = NeuronModel(genn_model, None, {}, {})
        self.add_input_logic(neuron_model, batch_size, population.shape)
        return neuron_model
