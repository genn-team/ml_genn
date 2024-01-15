from . import SurrogateGradient
from ..neurons import AdaptiveLeakyIntegrateFire, LeakyIntegrateFire, Neuron
from ..utils.model import WeightUpdateModel


class Boxcar(SurrogateGradient):
    def __init__(self, threshold: float = 0.5, value: float = 0.3):
        self.threshold = threshold
        self.value = value

    def add_to_weight_update(self, surrogate_var_name: str,
                             weight_update_model: WeightUpdateModel,
                             target_neuron: Neuron):
        # Add target threshold parameter to weight update model
        weight_update_model.add_param("Vthresh_post", "scalar", target_neuron.v_thresh)

        # Add postsynaptic variable references to
        # target membrane voltage and refractory time
        weight_update_model.add_post_neuron_var_ref("V_post", "scalar", "V")
        weight_update_model.add_post_neuron_var_ref("RefracTime_post",
                                                    "scalar", "RefracTime")

        # If target neuron is LIF
        if isinstance(target_neuron, LeakyIntegrateFire):
            # Add postsynaptic dynamics to calculate
            weight_update_model.append_post_dynamics_code(
                f"""
                if (RefracTime_post > 0.0) {{
                    {surrogate_var_name} = 0.0;
                }}
                else {{
                    {surrogate_var_name} = (fabs((V_post - Vthresh_post) / Vthresh_post) < {self.threshold}) ? {self.value} : 0.0;
                }}
                """)

        # Otherise, if it's ALIF, create weight update model with eProp ALIF
        elif isinstance(target_neuron, AdaptiveLeakyIntegrateFire):
            # Add target Beta parameter to weight update model
            weight_update_model.add_param("Beta_post", "scalar", target_neuron.beta)

            # Add postsynaptic variable references
            # to target adaptation variable
            weight_update_model.add_post_neuron_var_ref("A_post", "scalar", "A")

            # Add postsynaptic dynamics to calculate
            weight_update_model.append_post_dynamics_code(
                f"""
                if (RefracTime_post > 0.0) {{
                    {surrogate_var_name} = 0.0;
                }}
                else {{
                    {surrogate_var_name} = (fabs((V_post - (Vthresh_post + (Beta_post * A_post))) / Vthresh_post) < {self.threshold}) ? {self.value} : 0.0;
                }}
                """)
        else:
            raise NotImplementedError(f"Boxcar surrogate gradient "
                                      f"function doesn't support "
                                      f"{type(pop.neuron).__name__} neurons")
