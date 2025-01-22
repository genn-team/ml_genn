import numpy as np

from .initializer import Initializer
from ..utils.snippet import ConstantValueDescriptor, InitializerSnippet


# **NOTE** as long as scale_i is enabled on Exponential synapse, 
# mlGeNN neurons more-or-less perfectly match analytical solution
def _get_exponential_epsilon(tau_mem, tau_syn):
    epsilon_bar = tau_syn
    epsilon_hat = (tau_syn ** 2) / (2 * (tau_syn + tau_mem))

    return epsilon_bar, epsilon_hat

def _get_delta_epsilon(tau_mem):
    epsilon_bar = tau_mem
    epsilon_hat = tau_mem / 2
    
    return epsilon_bar, epsilon_hat

    
class FluctuationDrivenCentredNormal(Initializer):
    """Implements fluctuation-driven initialization with normally
    distributed, centred weights [Rossbroich2022]_. For use with
    non-Dalian networks of :class:`ml_genn.neurons.LeakyIntegrateFire` 
    neurons and :class:`ml_genn.synapses.Exponential` synapes
    
    Args:
        n:          Average number of synapses targetting postsynaptic neurons
        nu:         Estimated presynaptic firing rate [Hz]
        scale:      Proportion of membrane fluctuations caused by this
                    connection. 1 for purely feed-forward networks, 
                    typically 0.9 for feed-forward connections in recurrent
                    networks and 0.1 for the recurrent connections
        v_sigma:    Target standard deviation of postsynaptic neuron
                    membrane voltage
    """
    n = ConstantValueDescriptor()
    nu = ConstantValueDescriptor()
    scale = ConstantValueDescriptor()
    v_sigma = ConstantValueDescriptor()

    def __init__(self, n: float, nu: float, 
                 scale: float = 1.0, v_sigma: float = 1.0):
        super(FluctuationDrivenCentredNormal, self).__init__()
        
        self.n = n
        self.nu = nu
        self.scale = scale
        self.v_sigma = v_sigma

    def get_snippet(self, context) -> InitializerSnippet:
        from ..connection import Connection
        from ..neurons import LeakyIntegrateFire
        from ..synapses import Delta, Exponential
        
        # Ensure context is a connection
        if not isinstance(context, Connection):
            raise RuntimeError("FluctuationDrivenCentredNormal can only be "
                               "used to initialise variables associated "
                               "with connections e.g. weights")
        
        # Check target neuron model
        neuron = context.target().neuron
        if not isinstance(neuron, LeakyIntegrateFire):
            raise RuntimeError("FluctuationDrivenCentredNormal can only "
                               "be used with Connections which "
                               "target LeakyIntegrateFire neurons")

        # Calculate epsilon based on synapse type
        if isinstance(context.synapse, Delta):
            _, ehat = _get_delta_epsilon(neuron.tau_mem * 1E-3)
        elif isinstance(context.synapse, Exponential):
            _, ehat = _get_exponential_epsilon(neuron.tau_mem * 1E-3,
                                               context.synapse.tau * 1E-3)
        else:
            raise RuntimeError("FluctuationDrivenCentredNormal can only "
                               "be used with Connections using "
                               "Exponential or Delta synapses")
        
        # Calculate standard deviation of centred weight distribution
        sigma_w = np.sqrt(
            (self.scale * (neuron.v_thresh * self.v_sigma) ** 2) 
            / (self.n * self.nu * ehat))

        return InitializerSnippet("Normal", {"mean": 0.0, "sd": sigma_w})

    def __repr__(self):
        return (f"(FluctuationDrivenCentredNormal) N: {self.n}, "
                f"Nu: {self.nu}, V sigma: {self.v_sigma}")

