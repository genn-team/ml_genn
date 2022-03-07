from .neuron import Model, Neuron
from ..utils import InitValue, Value

class LeakyIntegrateFire(Neuron):
    def __init__(self, threshold=1.0, v=0.0, tau_mem=20.0, tau_refrac=None, 
                 relative_reset=True, integrate_during_refrac=True):
        super(LeakyIntegrateFire, self).__init__()
        
        self.threshold = Value(threshold)
        self.v = Value(v)
        self.tau_mem = Value(tau_mem)
        self.tau_refrac = Value(tau_refrac)
        self.relative_reset = relative_reset
        self.integrate_during_refrac = integrate_during_refrac
        
        if self.tau_mem.is_initializer:
            raise NotImplementedError("Leaky integrate and fire neuron model "
                                      "does not currently support tau_mem "
                                      "values specified using Initialiser objects")

    def get_model(self, population, dt):
        # Build basic model    
        genn_model = {
            "var_name_types": [("V", "scalar")],
            "param_name_types": [("Vthr", "scalar"), ("Alpha", "scalar")],
            "threshold_condition_code": "$(V) >= $(Vthr)",
            "is_auto_refractory_required": False}
        param_vals = {"Vthr": self.threshold, "Alpha": Value(np.exp(-dt / self.tau_mem.value))}
        var_vals = {"V": self.v}
        
        # Build reset code depending on whether 
        # reset should be relative or not
        if self.relative_refrac:
            genn_model["reset_code"] =\
                """
                $(V) -= $(Vthresh);
                """
        else:
            genn_model["reset_code"] =\
                """
                $(V) = 0.0;
                """

        # If neuron has refractory period
        if self.tau_refrac is not None:
            # Add state variable and parameter to control refractoryness
            genn_model["var_name_types"].append(("RefracTime", "scalar"))
            genn_model["param_name_types"].append(("TauRefrac", "scalar"))
            
            # Initialize
            param_vals["TauRefrac"] = self.tau_refrac
            var_vals["RefracTime"] = Value(0.0)
            
            # Build correct sim code depending on whether 
            # we should integrate during refractory period
            if self.integrate_during_refrac:
                genn_model["sim_code"] =\
                    """
                    $(V) = ($(Alpha) * $(V)) + $(Isyn);
                    if ($(RefracTime) > 0.0) {
                        $(RefracTime) -= DT;
                    }
                    """
            else:
                genn_model["sim_code"] =\
                    """
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
            genn_model["threshold_condition_code"] += " && $(RefracTime) <= 0.0"
        # Otherwise, build non-refractory sim-code
        else:
            genn_model["sim_code"] =\
                """
                $(V) = ($(Alpha) * $(V)) + $(Isyn);
                """
            
        # Return model
        return Model(genn_model, param_vals, var_vals)
