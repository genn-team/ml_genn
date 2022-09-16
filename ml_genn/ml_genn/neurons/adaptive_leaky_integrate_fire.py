import numpy as np

from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

from ..utils.value import is_value_initializer


class AdaptiveLeakyIntegrateFire(Neuron):
    v_thresh = ValueDescriptor()
    v_reset = ValueDescriptor()
    v = ValueDescriptor()
    a = ValueDescriptor()
    beta = ValueDescriptor()
    tau_mem = ValueDescriptor()
    tau_refrac = ValueDescriptor()
    tau_adapt = ValueDescriptor()

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, a: InitValue = 0.0, beta: InitValue = 0.0174,
                 tau_mem: InitValue = 20.0, tau_refrac: InitValue = None,
                 tau_adapt: InitValue = 2000.0, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, softmax: bool = False,
                 output=None):
        super(AdaptiveLeakyIntegrateFire, self).__init__(softmax, output)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.a = a
        self.beta = beta
        self.tau_mem = tau_mem
        self.tau_refrac = tau_refrac
        self.tau_adapt = tau_adapt
        self.relative_reset = relative_reset
        self.integrate_during_refrac = integrate_during_refrac

        if is_value_initializer(self.tau_mem):
            raise NotImplementedError("Adaptive leaky integrate and fire "
                                      "neuron model not currently support "
                                      "tau_mem values specified using "
                                      "Initialiser objects")
        if is_value_initializer(self.tau_adapt):
            raise NotImplementedError("Adaptive leaky integrate and fire "
                                      "neuron model not currently support "
                                      "tau_adapt values specified using "
                                      "Initialiser objects")

    def get_model(self, population, dt):
        # Build basic model
        genn_model = {
            "var_name_types": [("V", "scalar"), ("A", "scalar")],
            "param_name_types": [("Vthresh", "scalar"), ("Vreset", "scalar"),
                                 ("Alpha", "scalar"), ("Beta", "scalar"), 
                                 ("Rho", "scalar")],
            "threshold_condition_code": "$(V) >= ($(Vthresh) + ($(Beta) * $(A)))",
            "is_auto_refractory_required": False}
        param_vals = {"Vthresh": self.v_thresh, "Vreset": self.v_reset,
                      "Beta": self.beta, "Alpha": np.exp(-dt / self.tau_mem),
                      "Rho": np.exp(-dt / self.tau_adapt)}
        var_vals = {"V": self.v, "A": self.a}

        # Build reset code depending on whether
        # reset should be relative or not
        if self.relative_reset:
            genn_model["reset_code"] =\
                """
                $(V) -= ($(Vthresh) - $(Vreset));
                $(A) += 1.0;
                """
        else:
            genn_model["reset_code"] =\
                """
                $(V) = $(Vreset);
                $(A) += 1.0;
                """

        # If neuron has refractory period
        if self.tau_refrac is not None:
            # Add state variable and parameter to control refractoryness
            genn_model["var_name_types"].append(("RefracTime", "scalar"))
            genn_model["param_name_types"].append(("TauRefrac", "scalar"))

            # Initialize
            param_vals["TauRefrac"] = self.tau_refrac
            var_vals["RefracTime"] = 0.0

            # Build correct sim code depending on whether
            # we should integrate during refractory period
            if self.integrate_during_refrac:
                genn_model["sim_code"] =\
                    """
                    $(V) = ($(Alpha) * $(V)) + $(Isyn);
                    $(A) *= $(Rho);
                    if ($(RefracTime) > 0.0) {
                        $(RefracTime) -= DT;
                    }
                    """
            else:
                genn_model["sim_code"] =\
                    """
                    $(A) *= $(Rho);
                    if ($(RefracTime) > 0.0) {
                        $(RefracTime) -= DT;
                    }
                    else {
                        $(V) = ($(Alpha) * $(V)) + $(Isyn);
                    }
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
            genn_model["sim_code"] =\
                """
                $(V) = ($(Alpha) * $(V)) + $(Isyn);
                $(A) *= $(Rho);
                """

        # Return model
        return NeuronModel(genn_model, "V", param_vals, var_vals)
