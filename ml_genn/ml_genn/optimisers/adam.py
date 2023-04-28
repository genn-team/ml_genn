from pygenn.genn_wrapper.Models import (VarAccessMode_READ_ONLY,
                                        VarAccessMode_READ_WRITE)
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


genn_model = {
    "var_name_types": [("M", "scalar"), ("V", "scalar")],
    "param_name_types": [("Beta1", "scalar"), ("Beta2", "scalar"),
                         ("Epsilon", "scalar")],
    "extra_global_params": [("Alpha", "scalar"),
                            ("MomentScale1", "scalar"),
                            ("MomentScale2", "scalar")],
    "var_refs": [("Gradient", "scalar", VarAccessMode_READ_ONLY),
                 ("Variable", "scalar")],
    "update_code":
        """
        // Update biased first moment estimate
        $(M) = ($(Beta1) * $(M)) + ((1.0 - $(Beta1)) * $(Gradient));

        // Update biased second moment estimate
        $(V) = ($(Beta2) * $(V)) + ((1.0 - $(Beta2)) * $(Gradient) * $(Gradient));

        // Add gradient to variable, scaled by learning rate
        $(Variable) -= ($(Alpha) * $(M) * $(MomentScale1)) / (sqrt($(V) * $(MomentScale2)) + $(Epsilon));
        """}


class Adam(Optimiser):
    alpha = ConstantValueDescriptor()
    beta1 = ConstantValueDescriptor()
    beta2 = ConstantValueDescriptor()
    epsilon = ConstantValueDescriptor()

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def set_step(self, genn_cu, step):
        assert step >= 0
        moment_scale_1 = 1.0 / (1.0 - (self.beta1 ** (step + 1)))
        moment_scale_2 = 1.0 / (1.0 - (self.beta2 ** (step + 1)))

        genn_cu.extra_global_params["Alpha"].view[:] = self.alpha
        genn_cu.extra_global_params["MomentScale1"].view[:] = moment_scale_1
        genn_cu.extra_global_params["MomentScale2"].view[:] = moment_scale_2

    def get_model(self, gradient_ref, var_ref, zero_gradient: bool):
        model = CustomUpdateModel(
            deepcopy(genn_model),
            {"Beta1": self.beta1, "Beta2": self.beta2,
             "Epsilon": self.epsilon},
            {"M": 0.0, "V": 0.0},
            {"Gradient": gradient_ref, "Variable": var_ref},
            {"Alpha": self.alpha, "FirstMomentScale": 0.0,
             "SecondMomentScale": 0.0})

        # If a optimiser than automatically zeros
        # gradients should be provided
        if zero_gradient:
            # Change variable access model of gradient to read-write
            model.set_var_ref_access_mode("Gradient",
                                          VarAccessMode_READ_WRITE)

            # Add update code to zero the gradient
            model.append_update_code(
                """
                // Zero gradient
                $(Gradient) = 0.0;
                """)

        # Return model
        return model
