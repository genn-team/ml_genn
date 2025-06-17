from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING
from .neuron import Neuron
from ..utils.auto_model import AutoNeuronModel, Variables
from ..utils.model import NeuronModel
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Population

from copy import copy

class UserNeuron(Neuron):
    """A wrapper to allow users to easily define their own neuron models.
    
    Args:
        vars:               Dictionary specifying the dynamics and jumps for
                            each state variable as expressions, 
                            parsable by sympy
        output_var_name:    Name of variable used for output
        threshold:          Expression in sympy format
        param_vals:         Initial values for all parameters
        var_vals:           Initial values for all state variables
        readout:            Type of readout to attach to this neuron's output variable
    """

    def __init__(self, vars: Variable, output_var_name: str,
                 threshold: Optional[str] = None,
                 param_vals: MutableMapping[str, InitValue] = {},
                 var_vals: MutableMapping[str, InitValue] = {},
                 solver: str = "exponential_euler",
                 sub_steps: int = 1,
                 readout=None, **kwargs):
        super(UserNeuron, self).__init__(readout, **kwargs)

        self.vars = vars
        self.output_var_name = output_var_name
        self.threshold = threshold
        self.param_vals = param_vals
        self.var_vals = var_vals
        self.solver = solver
        self.sub_steps = sub_steps

    def get_model(self, population: Population, dt: float,
                  batch_size: int) -> Union[AutoNeuronModel, NeuronModel]:
        return AutoNeuronModel({"vars": copy(self.vars), 
                                "threshold": self.threshold},
                               self.output_var_name, self.param_vals,
                               self.var_vals, self.solver, self.sub_steps)
