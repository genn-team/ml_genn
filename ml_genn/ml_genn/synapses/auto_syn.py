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
    def __init__(self, vars: list, params: list, ode:dict):
        super(AutoSyn, self).__init__()

        self.vars = vars
        self.params = params
        self.ode = ode
        self.genn_model = {}
        # add vars to genn_model. Assume all are scalars for now
        genn_vars = []
        for var in self.vars:
            genn_vars.append((var, "scalar"))
        self.genn_model["vars"] = genn_vars

        # add params to genn_model. Assume are scalars for now
        genn_params = []
        for p in self.params:
            genn_params.append((p, "scalar"))
        self.genn_model["params"] = genn_params
        # sim_code will be added in the compiler        
       
    def get_model(self, connection: Connection,
                  dt: float, batch_size: int) -> SynapseModel:
        return SynapseModel.from_val_descriptors(self.genn_model, self, dt)
