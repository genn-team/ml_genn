from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from .synapse import Synapse
from ..utils.model import SynapseModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Connection

from ..utils.value import is_value_initializer

from ..utils.decorators import network_default_params

class AutoSyn(Synapse):
    """Synapse model with auto-generated equations

    Args:
       vars: list of variable names, including mandatory exatly one I
       params: list of parameter names
       ode: dict of differential equations including one for I
    """

    @network_default_params
    def __init__(self, vars: list, params: list, ode: dict, jumps: dict, w_name: str, inject_current: str, solver="exponential_euler"):
        super(AutoSyn, self).__init__()

        self.vars = vars
        self.varnames = [ var[0] for var in vars]
        self.var_vals = { var[0]: var[2] for var in vars }
        self.params = params
        self.pnames = [ p[0] for p in params ]
        self.param_vals = {p[0]: p[2] for p in params }
        self.ode = ode
        self.jumps = jumps
        self.w_name = w_name
        self.inject_current = inject_current
        self.solver = solver
        self.lbd_ode = {}
        self.genn_model = {}
        # add vars to genn_model. Assume all are scalars for now
        genn_vars = []
        for var in self.vars:
            genn_vars.append((var[0], var[1]))
        self.genn_model["vars"] = genn_vars

        # add params to genn_model. Assume are scalars for now
        genn_params = []
        for p in self.params:
            genn_params.append((p[0], p[1]))
        self.genn_model["params"] = genn_params
        self.genn_model["sim_code"] = ""
        self.dl_dt = {}
        self.gradient_update_code = ""
        self.add_to_pre = {}
        self.post_var_refs = {}
        print(self.genn_model)
        
       
    def get_model(self, connection: Connection,
                  dt: float, batch_size: int) -> SynapseModel:
        return SynapseModel(self.genn_model, param_vals=self.param_vals,var_vals=self.var_vals)
