from __future__ import annotations

import numpy as np

from typing import Optional, TYPE_CHECKING
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population

from ..utils.decorators import network_default_params

class AdaptiveLeakyIntegrateFire(Neuron):
    """A leaky-integrate and fire neuron with an adaptive firing threshold
    as described by [Bellec2018]_.
    
    Args:
        v_thresh:                   Membrane voltage firing threshold
        v_reset:                    After a spike is emitted, this value is
                                    *subtracted* from the membrane voltage
                                    ``v`` if ``relative_reset`` is ``True``.
                                    Otherwise, if ``relative_reset`` is 
                                    ``False``, the membrane voltage is set to
                                    this value.
        v:                          Initial value of membrane voltage
        a:                          Initial value of adaptation
        beta:                       Strength of adaptation
        tau_mem:                    Time constant of membrane voltage [ms]
        tau_refrac:                 Duration of refractory period [ms]
        tau_adapt:                  Time constant of adaptation [ms]
        relative_reset:             How is ``v`` reset after a spike?
        integrate_during_refrac:    Should ``v`` continue to integrate inputs
                                    during refractory period?
        readout:                    Type of readout to attach to this
                                    neuron's output variable
    """
    
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")
    a = ValueDescriptor("A")
    beta = ValueDescriptor("Beta")
    tau_mem = ValueDescriptor(("Alpha", lambda val, dt: np.exp(-dt / val)))
    tau_refrac = ValueDescriptor("TauRefrac")
    tau_adapt = ValueDescriptor(("Rho", lambda val, dt: np.exp(-dt / val)))
    
    @network_default_params
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, a: InitValue = 0.0, beta: InitValue = 0.0174,
                 tau_mem: InitValue = 20.0, tau_refrac: InitValue = None,
                 tau_adapt: InitValue = 2000.0, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, readout=None):
        super(AdaptiveLeakyIntegrateFire, self).__init__(readout)

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

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        # Build basic model
        genn_model = {
            "vars": [("V", "scalar"), ("A", "scalar")],
            "params": [("Vthresh", "scalar"), ("Vreset", "scalar"),
                       ("Alpha", "scalar"), ("Beta", "scalar"), 
                       ("Rho", "scalar")],
            "threshold_condition_code": "V >= (Vthresh + (Beta * A))"}

        # Build reset code depending on whether
        # reset should be relative or not
        if self.relative_reset:
            genn_model["reset_code"] =\
                """
                V -= (Vthresh - Vreset);
                A += 1.0;
                """
        else:
            genn_model["reset_code"] =\
                """
                V = Vreset;
                A += 1.0;
                """

        # If neuron has refractory period
        if self.tau_refrac is not None:
            # Add state variable and parameter to control refractoryness
            genn_model["vars"].append(("RefracTime", "scalar"))
            genn_model["params"].append(("TauRefrac", "scalar"))

            # Build correct sim code depending on whether
            # we should integrate during refractory period
            if self.integrate_during_refrac:
                genn_model["sim_code"] =\
                    """
                    V = (Alpha * V) + Isyn;
                    A *= Rho;
                    if (RefracTime > 0.0) {
                        RefracTime -= dt;
                    }
                    """
            else:
                genn_model["sim_code"] =\
                    """
                    A *= Rho;
                    if (RefracTime > 0.0) {
                        RefracTime -= dt;
                    }
                    else {
                        V = (Alpha * V) + Isyn;
                    }
                    """

            # Add refractory period initialisation to reset code
            genn_model["reset_code"] +=\
                """
                RefracTime = TauRefrac;
                """

            # Add refractory check to threshold condition
            genn_model["threshold_condition_code"] +=\
                " && RefracTime <= 0.0"
        # Otherwise, build non-refractory sim-code
        else:
            genn_model["sim_code"] =\
                """
                V = (Alpha * V) + Isyn;
                A *= Rho;
                """

        # Return model
        var_vals = {} if self.tau_refrac is None else {"RefracTime": 0.0}
        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt,
                                                var_vals=var_vals)
