import numpy as np

from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor


class LeakyIntegrateFire(Neuron):
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")
    tau_mem = ValueDescriptor("Alpha", lambda val, dt: np.exp(-dt / val))
    tau_refrac = ValueDescriptor("TauRefrac")

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, tau_mem: InitValue = 20.0,
                 tau_refrac: InitValue = None, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, scale_i: bool = False,
                 softmax: bool = False, readout=None):
        super(LeakyIntegrateFire, self).__init__(softmax, readout)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.tau_mem = tau_mem
        self.tau_refrac = tau_refrac
        self.relative_reset = relative_reset
        self.integrate_during_refrac = integrate_during_refrac
        self.scale_i = scale_i

    def get_model(self, population, dt):
        # Build basic model
        genn_model = {
            "var_name_types": [("V", "scalar")],
            "param_name_types": [("Vthresh", "scalar"), ("Vreset", "scalar"),
                                 ("Alpha", "scalar")],
            "threshold_condition_code": "$(V) >= $(Vthresh)",
            "is_auto_refractory_required": False}

        # Build reset code depending on whether
        # reset should be relative or not
        if self.relative_reset:
            genn_model["reset_code"] =\
                """
                $(V) -= ($(Vthresh) - $(Vreset));
                """
        else:
            genn_model["reset_code"] =\
                """
                $(V) = $(Vreset);
                """
        # Define integration code based on whether I should be scaled
        if self.scale_i:
            v_update = "$(V) = ($(Alpha) * $(V)) + ((1.0 - $(Alpha)) * $(Isyn));"
        else:
            v_update = "$(V) = ($(Alpha) * $(V)) + $(Isyn);"

        # If neuron has refractory period
        if self.tau_refrac is not None:
            # Add state variable and parameter to control refractoryness
            genn_model["var_name_types"].append(("RefracTime", "scalar"))
            genn_model["param_name_types"].append(("TauRefrac", "scalar"))

            
            # Build correct sim code depending on whether
            # we should integrate during refractory period
            if self.integrate_during_refrac:
                genn_model["sim_code"] =\
                    f"""
                    {v_update}
                    if ($(RefracTime) > 0.0) {{
                        $(RefracTime) -= DT;
                    }}
                    """
            else:
                genn_model["sim_code"] =\
                    f"""
                    if ($(RefracTime) > 0.0) {{
                        $(RefracTime) -= DT;
                    }}
                    else {{
                        {v_update}
                    }}
                    """

            # Add refractory period initialisation to reset code
            genn_model["reset_code"] +=\
                """
                $(RefracTime) = $(TauRefrac);
                """

            # Add refractory check to threshold condition
            genn_model["threshold_condition_code"] +=\
                " && $(RefracTime) <= 0.0"
        # Otherwise, build non-refractory sim-code
        else:
            genn_model["sim_code"] = v_update

        # Return model
        var_vals = {} if self.tau_refrac is None else {"RefracTime": 0.0}
        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt,
                                                var_vals=var_vals)
