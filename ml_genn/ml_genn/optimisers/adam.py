from typing import Optional, Tuple
from pygenn import VarAccessMode
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


genn_model = {
    "vars": [("M", "scalar"), ("V", "scalar")],
    "params": [("Beta1", "scalar"), ("Beta2", "scalar"),
               ("Epsilon", "scalar"), ("Alpha", "scalar"),
               ("MomentScale1", "scalar"), ("MomentScale2", "scalar")],
    "var_refs": [("Gradient", "scalar", VarAccessMode.READ_ONLY),
                 ("Variable", "scalar")],
    "update_code":
        """
        // Update biased first moment estimate
        M = (Beta1 * M) + ((1.0 - Beta1) * Gradient);

        // Update biased second moment estimate
        V = (Beta2 * V) + ((1.0 - Beta2) * Gradient * Gradient);

        // Add gradient to variable, scaled by learning rate
        Variable -= (Alpha * M * MomentScale1) / (sqrt(V * MomentScale2) + Epsilon);
        """}


class Adam(Optimiser):
    """Optimizer that implements the Adam algorithm [Kingma2014]_.
    Adam optimization is a stochastic gradient descent method that 
    is based on adaptive estimation of first-order and second-order moments.
    
    Args:
        alpha:      Learning rate
        beta1:      The exponential decay rate for the 1st moment estimates.
        beta2:      The exponential decay rate for the 2nd moment estimates.
        epsilon:    A small constant for numerical stability. This is
                    the epsilon in Algorithm 1 of the [Kingma2014]_
    """
    alpha = ConstantValueDescriptor()
    beta1 = ConstantValueDescriptor()
    beta2 = ConstantValueDescriptor()
    epsilon = ConstantValueDescriptor()

    def __init__(self, alpha: float = 0.001, beta1 : float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def set_step(self, genn_cu, step):
        assert step >= 0
        moment_scale_1 = 1.0 / (1.0 - (self.beta1 ** (step + 1)))
        moment_scale_2 = 1.0 / (1.0 - (self.beta2 ** (step + 1)))

        genn_cu.set_dynamic_param_value("Alpha", self.alpha)
        genn_cu.set_dynamic_param_value("MomentScale1", moment_scale_1)
        genn_cu.set_dynamic_param_value("MomentScale2", moment_scale_2)


    def get_model(self, gradient_ref, var_ref, zero_gradient: bool,
                  clamp_var: Optional[Tuple[float, float]] = None) -> CustomUpdateModel:
                  positive_sign_change_egp_ref=None, 
                  negative_sign_change_egp_ref=None):
        model = CustomUpdateModel(
            deepcopy(genn_model),
            {"Beta1": self.beta1, "Beta2": self.beta2,
             "Epsilon": self.epsilon, "Alpha": self.alpha, 
             "MomentScale1": 0.0, "MomentScale2": 0.0},
            {"M": 0.0, "V": 0.0},
            {"Gradient": gradient_ref, "Variable": var_ref})

        # Make parameters dynamic
        model.set_param_dynamic("Alpha")
        model.set_param_dynamic("MomentScale1")
        model.set_param_dynamic("MomentScale2")

        # If a optimiser than automatically zeros
        # gradients should be provided
        # **THINK** this is generic across all optimisers like readout-adding
        if zero_gradient:
            # Change variable access model of gradient to read-write
            model.set_var_ref_access_mode("Gradient",
                                          VarAccessMode.READ_WRITE)

            # Add update code to zero the gradient
            model.append_update_code(
                """
                // Zero gradient
                Gradient = 0.0;
                """)

        # If variable should be clamped
        # **THINK** this is generic across all optimisers like readout-adding
        if clamp_var is not None:
            # Add minimum and maximum parameters
            model.add_param("VariableMin", "scalar", clamp_var[0])
            model.add_param("VariableMax", "scalar", clamp_var[1])
            
            # Add update code to clamp variable
            model.append_update_code(
                """
                // Clamp variable
                Variable = fmax(VariableMin, fmin(VariableMax, Variable));
                """)

        # Check we're not tracking positive AND negative
        assert (positive_sign_change_egp_ref is None 
                or negative_sign_change_egp_ref is None)

        if positive_sign_change_egp_ref is not None:
            # Add EGP ref
            model.add_egp_ref("SignChange", "uint32_t*",
                              positive_sign_change_egp_ref)
            
            # Add update code to set bit if variable goes positive
            model.append_update_code(
                """
                if(Variable > 0.0) {
                    atomic_or(SignChange + (id_syn / 32), 1 << (id_syn % 32));
                }
                """)
        
        if negative_sign_change_egp_ref is not None:
            # Add EGP ref
            model.add_egp_ref("SignChange", "uint32_t*",
                              negative_sign_change_egp_ref)
            
            # Add update code to set bit if variable goes negative
            model.append_update_code(
                """
                if(Variable < 0.0) {
                    atomic_or(SignChange + (id_syn / 32), 1 << (id_syn % 32));
                }
                """)

        # Return model
        return model
