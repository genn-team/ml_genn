from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING
from .synapse import Synapse
from ..utils.auto_model import AutoSynapseModel, Variables
from ..utils.model import SynapseModel
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Population

from copy import copy

class UserSynapse(Synapse):
    """A wrapper to allow users to easily define their own synapse models.
    
    Args:
        vars:           Dictionary specifying the dynamics and jumps for
                        each state variable as expressions, parsable by sympy
        inject_current: Expression in sympy format specifcying
        param_vals:     Initial values for all parameters
        var_vals:       Initial values for all state variables
    """

    def __init__(self, vars: Variable, inject_current: str,
                 param_vals: MutableMapping[str, InitValue] = {},
                 var_vals: MutableMapping[str, InitValue] = {},
                 solver: str = "exponential_euler",
                 sub_steps: int = 1):
        super(UserSynapse, self).__init__()

        self.vars = vars
        self.inject_current = inject_current
        self.param_vals = param_vals
        self.var_vals = var_vals
        self.solver = solver
        self.sub_steps = sub_steps

    def get_model(self, connection: Connection, dt: float,
                  batch_size: int) -> Union[AutoSynapseModel, SynapseModel]:
        return AutoSynapseModel({"vars": copy(self.vars), 
                                 "inject_current": self.inject_current},
                                self.param_vals, self.var_vals, self.solver,
                                self.sub_steps)
