import numpy as np

from .initializer import Initializer
from ..utils.snippet import ConstantValueDescriptor, InitializerSnippet


# **NOTE** as long as scale_i is enabled on Exponential synapse, 
# mlGeNN neurons more-or-less perfectly match analytical solution
def _get_epsilon(tau_mem, tau_syn):
    epsilon_bar = tau_syn
    epsilon_hat = (tau_syn ** 2) / (2 * (tau_syn + tau_mem))

    return epsilon_bar, epsilon_hat


class FluctuationDrivenCentredNormal(Initializer):
    """Implements flucutation-driven initialization as described in [Rossbroich2022]_.
    
    Args:
        n:          Average fan-in of connection
        nu:         Estimated presynaptic firing rate [Hz]
        scale:      ss
        v_sigma:    Target standard deviation of postsynaptic neuron
                    membrane voltage
        tau_mem:    Membrane time constant of postsynaptic neuron [ms]
        tau_syn:    Synaptic time constant of posynaptic model [ms]
        v_thresh:   Membrane voltage firing threshold for postsynaptic neuron
    """
    n = ConstantValueDescriptor()
    nu = ConstantValueDescriptor()
    scale = ConstantValueDescriptor()
    v_sigma = ConstantValueDescriptor()
    tau_mem = ConstantValueDescriptor()
    tau_syn = ConstantValueDescriptor()
    v_thresh = ConstantValueDescriptor()

    def __init__(self, n: float, nu: float, scale: float = 1.0,
                 v_sigma: float = 1.0, tau_mem: float = 20.0, 
                 tau_syn: float = 5.0, v_thresh: float = 1.0):
        super(FluctuationDrivenCentredNormal, self).__init__()
        
        self.n = n
        self.nu = nu
        self.scale = scale
        self.v_sigma = v_sigma
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_thresh = v_thresh

    def get_snippet(self) -> InitializerSnippet:
        # Calculate epsilon
        _, ehat = _get_epsilon(self.tau_mem * 1E-3, self.tau_syn * 1E-3)
        
        # Calculate standard deviation of centred weight distribution
        sigma_w = np.sqrt(
            (self.scale * (self.v_thresh * self.v_sigma) ** 2) 
            / (self.n * self.nu * ehat))
        
        print(sigma_w)
        return InitializerSnippet("Normal", {"mean": 0.0, "sd": sigma_w})

    def __repr__(self):
        return (f"(FluctuationDrivenCentredNormal) N: {self.n}, "
                f"Nu: {self.nu}, V sigma: {self.v_sigma}, "
                f"Tau mem: {self.tau_mem}, Tau syn: {self.tau_syn}, "
                f"V thresh: {self.v_thresh}")

