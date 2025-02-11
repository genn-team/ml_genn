import numpy as np

from typing import Any, MutableMapping
from .model import NeuronModel, SynapseModel
from .value import Value

from ..utils.value import get_auto_values

class AutoModel:
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: MutableMapping[str, Value] = {},
                 var_vals: MutableMapping[str, Value] = {}):
        self.model = model

        self.param_vals = param_vals
        self.var_vals = var_vals
    
    def add_param(self, name: str, value: Value):
        assert not self.has_param(name)
        self._add_to_list("params", (name, type))
        self.param_vals[name] = value

    def add_var(self, name: str, value: Value,
                ode: str, jump: str):
        assert not self.has_var(name)
        self._add_to_list("vars", (name, ode, jump))
        self.var_vals[name] = value
    
    def get_model_vars(self, var_type: str = "scalar"):
        return [(v[0], var_type) for v in self.model["vars"]]
    
    def get_model_params(self, param_type: str = "scalar"):
        return [(p[0], param_type) for p in self.model["params"]]
    
    def get_model_jump_code(self):
        jumps = [f"{v[0]} = {v[2]};" for v in self.model["vars"]
                 if v[2] is not None and v[2] != v[0]]
        return "\n".join(jumps)
    
    @staticmethod
    def from_val_descriptors(model, inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, model.get("vars", {}).keys(),
                                       var_vals, param_vals)
        return AutoModel(model, param_vals, var_vals)
        
    
        
    