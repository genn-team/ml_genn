from .encoder import Encoder, Model
from ..utils import Value

class BinarySpike(Encoder):
    def __init__(self, signed_spikes=False):
        super(BinarySpike, self).__init__()

        self.signed_spikes = signed_spikes

    def get_model(self, population, dt):
        genn_model = {
            "var_name_types": [("Input", "scalar")],
            "sim_code": 
                """
                const bool spike = $(Input) != 0.0;
                """,
            "threshold_condition_code":
                """
                $(Input) > 0.0 && spike
                """,
            is_auto_refractory_required:False}
        
        # If signed spikes are enabled, add negative threshold condition
        if self.signed_spikes:
            genn_model["negative_threshold_condition_code"] =\
                """
                $(Input_pre) < 0.0 && spike
                """
        
        return Model(genn_model, {}, {"Input": Value(0.0)})
