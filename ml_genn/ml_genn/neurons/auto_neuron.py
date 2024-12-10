from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor
from ..auto_tools import *

if TYPE_CHECKING:
    from .. import Population
    

class AutoNeuron(Neuron):
    """A neuron with auto-generated equations
    
    Args:
        vars: list of variable names, not including I
        params: list of parameter names
        ode: dict of differential equations including "I" on rhs but not including I equation
        threshold: threshold condition 
        reset: dict of reset expressions (after own spike)
        readout:    Type of readout to attach to this
                    neuron's output variable
    """

    def __init__(self, vars: list, params: list,
                 ode: dict, threshold: str, reset: dict, readout=None, solver="exponential_euler", **kwargs):
        super(AutoNeuron, self).__init__(readout, **kwargs)

        self.vars = vars
        self.varnames = [ var[0] for var in vars ]
        self.params = params
        self.ode = ode
        self.threshold = threshold
        self.reset = reset
        self.solver = solver
        
        self.genn_model = {}
        # add vars to genn_model. Assume all are scalars for now
        genn_vars = []
        for var in self.vars:
            genn_vars.append((var[0], var[1]))
        self.genn_model["vars"] = genn_vars

        # add params to genn_model. Assume are scalars for now
        genn_params = []
        for p in self.params:
            genn_params.append((p, "scalar"))
        self.genn_model["params"] = genn_params

        self.genn_model["threshold_condition_code"] = f"{self.threshold} == 0"

        resets = []
        for var in self.varnames:
            if var in self.reset and var != self.reset[var]:
                resets.append(f"{var} = {reset[var]};")
            self.genn_model["reset_code"] = "\n".join(resets)

        # updates for forward pass
        sym = get_symbols(self.varnames, self.params)
        sym["I"] = sympy.Symbol("I")
        dt = sympy.Symbol("dt")
        self.dx_dt, clines = solve_ode(self.varnames, sym, self.ode, dt, self.solver)
        self.genn_model["sim_code"] = "\n".join(clines)
        print(self.genn_model)
        
    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        return NeuronModel.from_val_descriptors(self.genn_model, "V", self, dt)

