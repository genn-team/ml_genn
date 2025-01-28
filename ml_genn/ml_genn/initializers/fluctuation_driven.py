import numpy as np

from .initializer import Initializer
from ..utils.snippet import ConstantValueDescriptor, InitializerSnippet


# Calculate epsilon based on synapse type
# **NOTE** as long as scale_i is enabled on Exponential synapse, 
# mlGeNN neurons more-or-less perfectly match analytical solution
def _get_epsilon(synapse, tau_mem):
    from ..synapses import Delta, Exponential
    
    tau_mem_s = tau_mem * 1E-3
    if isinstance(synapse, Delta):
        epsilon_bar = tau_mem_s
        epsilon_hat = tau_mem_s / 2
    elif isinstance(synapse, Exponential):
        tau_syn_s = synapse.tau * 1E-3
        epsilon_bar = tau_syn_s
        epsilon_hat = (tau_syn_s ** 2) / (2 * (tau_syn_s + tau_mem_s))
    else:
        raise RuntimeError("Fluctuation-driven initializatio can only "
                           "be used with Connections using "
                           "Exponential or Delta synapses")
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
        
        # **TODO** could detect whether this connection is recurrent or not etc and provide alpha directly
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
        _, ehat = _get_epsilon(context.synapse, neuron.tau_mem)
        
        # Calculate standard deviation of centred weight distribution
        sigma_w = np.sqrt(
            (self.scale * (neuron.v_thresh * self.v_sigma)**2) 
            / (self.n * self.nu * ehat))

        return InitializerSnippet("Normal", {"mean": 0.0, "sd": sigma_w})

    def __repr__(self):
        return (f"(FluctuationDrivenCentredNormal) N: {self.n}, "
                f"Nu: {self.nu}, V sigma: {self.v_sigma}")


class FluctuationDrivenExponential(Initializer):
    """Implements fluctuation-driven initialization with exponentially
     distributed weights [Rossbroich2022]_. For use with Dalian
     networks of :class:`ml_genn.neurons.LeakyIntegrateFire` 
     neurons and :class:`ml_genn.synapses.Exponential` synapes
    
    Args:
        n_exc_ff:       Average number of feedforward excitatory synapses
                        targetting postsynaptic neurons
        n_exc_rec:      Average number of recurrent excitatory synapses
                        targetting postsynaptic neurons
        n_inh:          Average number of inhibitory synapses targetting
                        postsynaptic neurons
        nu:             Estimated presynaptic firing rate [Hz]
        alpha:          ss
        v_sigma:        Target standard deviation of postsynaptic neuron
                        membrane voltage
    """
    n_exc_ff = ConstantValueDescriptor()
    n_exc_rec = ConstantValueDescriptor()
    n_inh = ConstantValueDescriptor()
    nu = ConstantValueDescriptor()
    alpha = ConstantValueDescriptor()
    v_sigma = ConstantValueDescriptor()

    def __init__(self, n_exc_ff: float, n_exc_rec: float, n_inh: float,
                 nu: float, alpha: float = 0.9, v_sigma: float = 1.0):
        super(FluctuationDrivenExponential, self).__init__()
        
        self.n_exc_ff = n_exc_ff
        self.n_exc_rec = n_exc_rec
        self.n_inh = n_inh
        self.nu = nu
        self.alpha = alpha
        self.v_sigma = v_sigma

    def get_snippet(self, context) -> InitializerSnippet:
        from ..connection import Connection, Polarity
        from ..neurons import LeakyIntegrateFire
        from ..synapses import Delta, Exponential
        
        # Ensure context is a connection
        if not isinstance(context, Connection):
            raise RuntimeError("FluctuationDrivenExponential can only be "
                               "used to initialise variables associated "
                               "with connections e.g. weights")
        
        # Check target neuron model
        target = context.target()
        neuron = target.neuron
        if not isinstance(neuron, LeakyIntegrateFire):
            raise RuntimeError("FluctuationDrivenExponential can only "
                               "be used with Connections which "
                               "target LeakyIntegrateFire neurons")
        
        # Get all inhibitory and excitatory 
        # connections going into target neuron
        inh_cons = [c() for c in target.incoming_connections 
                    if c().polarity == Polarity.INHIBITORY]
        exc_cons = [c() for c in target.incoming_connections 
                    if c().polarity == Polarity.EXCITATORY]
                
        # Check that there is at least one excitatory and one inhibitory connection
        if len(inh_cons) == 0:
            raise RuntimeError("FluctuationDrivenExponential requires "
                               "each population to have at least one "
                               "incoming inhibitory connection")
        if len(exc_cons) == 0:
            raise RuntimeError("FluctuationDrivenExponential requires "
                               "each population to have at least one "
                               "incoming excitatory connection")

        # Get epsilon for each excitatory and inhibitory connection
        ebar_exc, ehat_exc = zip(*(_get_epsilon(c.synapse, neuron.tau_mem)
                                   for c in exc_cons))
        ebar_inh, ehat_inh = zip(*(_get_epsilon(c.synapse, neuron.tau_mem)
                                   for c in inh_cons))
        
        print(f"ebar_exc: {ebar_exc[0]}, ehat_exc: {ehat_exc[0]}, ebar_inh: {ebar_inh[0]}, ehat_inh: {ehat_inh[0]}")
        # Check all inhibitory and excitatory synapses have the same dynamics
        if not np.allclose(ebar_exc[0], ebar_exc):
            raise RuntimeError("FluctuationDrivenExponential requires each "
                               "excitatory synapse to have same dynamics")
        
        if not np.allclose(ebar_inh[0], ebar_inh):
            raise RuntimeError("FluctuationDrivenExponential requires each "
                               "inhibitory synapse to have same dynamics")
        
        # Further split excitatory connections into feedforward and recurrent
        exc_rec_cons = [c for c in exc_cons if c.target() == c.source()]
        exc_ff_cons = [c for c in exc_cons if c.target() != c.source()]
        
        # Calculate average total number of excitatory synapses
        n_exc_total = self.n_exc_ff + self.n_exc_rec
        
        # If there is recurrent excitation, use alpha scaling factor
        if len(exc_rec_cons) >= 1:
            delta_rec = np.sqrt(
                (self.alpha * self.n_exc_rec) 
                / (self.n_exc_ff - self.alpha * self.n_exc_ff))
            delta_ei = (delta_rec * ebar_inh[0] * self.n_inh * self.nu) / (
                delta_rec * ebar_exc[0] * self.n_exc_ff * self.nu
                + ebar_exc[0] * self.n_exc_rec * self.nu)

            lambda_exc_ff = (
                np.sqrt(2)
                * np.sqrt(self.nu)
                * np.sqrt(
                    delta_ei**2 * ehat_exc[0] * self.n_exc_rec
                    + delta_rec**2
                    * (
                        self.n_exc_ff * delta_ei**2 * ehat_exc[0]
                        + ehat_inh[0] * self.n_inh
                    )
                )
                / (neuron.v_thresh * delta_ei * delta_rec * self.v_sigma))

            if context.polarity == Polarity.INHIBITORY:
                print("Rec INHIB")
                lambda_w = -lambda_exc_ff * delta_ei
            else:
                if context.target() == context.source():
                    print("Rec Rec")
                    lambda_w = lambda_exc_ff * delta_rec
                else:
                    print("Rec FF")
                    lambda_w = lambda_exc_ff

        # If not, scale automatically by number of synapses
        else:
            delta_ei = (self.n_inh * ebar_inh[0] * self.nu) / (
                        n_exc_total * ebar_exc[0] * self.nu)
            lambda_exc_ff = (
                np.sqrt(2)
                * np.sqrt(
                    delta_ei**2 * n_exc_total * self.nu * ehat_exc[0]
                    + self.n_inh * self.nu * ehat_inh[0]
                )
                / (delta_ei * neuron.v_thresh * self.v_sigma))

            if context.polarity == Polarity.INHIBITORY:
                print("FF INHIB")
                lambda_w = -lambda_exc_ff * delta_ei
            else:
                print("FF Exc")
                lambda_w = lambda_exc_ff

        print(lambda_w)
        return InitializerSnippet("Exponential", {"lambda": lambda_w})

    def __repr__(self):
        return (f"(FluctuationDrivenExponential) N exc ff: {self.n_exc_ff}, "
                f"N exc rec: {self.n_exc_rec}, N inh: {self.n_inh}, "
                f"Nu: {self.nu}, Alpha: {self.alpha}, V sigma: {self.v_sigma}")
