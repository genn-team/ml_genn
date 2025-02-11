import numpy as np
import sympy

from typing import Any, MutableMapping
from .model import NeuronModel, SynapseModel
from .value import Value

from itertools import chain
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

    def get_vars(self, var_type: str = "scalar"):
        return [(n, var_type) for n in self.var_vals.keys()]
    
    def get_params(self, param_type: str = "scalar"):
        return [(n, param_type) for n in self.param_vals.keys()]

    def get_jump_code(self):
        jumps = [f"{v[0]} = {v[2]};" for v in self.model["vars"]
                 if v[2] is not None and v[2] != v[0]]
        return "\n".join(jumps)
    
    def get_symbols(self):
        return {n: sympy.Symbol(n) for n in chain(self.var_vals.keys(),
                                                  self.param_vals.keys())}
    
    def parse_odes(self, symbols):
        return {n: sympy.parse_expr(v[0], local_dict=symbols)
                for n, v in self.model["vars"].items()
                if v[0] is not None}
    
        
class AutoNeuronModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any], output_var_name: str,
                 param_vals: MutableMapping[str, Value] = {},
                 var_vals: MutableMapping[str, Value] = {}):
        super(AutoNeuronModel, self).__init__(model, param_vals, var_vals)

        self.output_var_name = output_var_name

    def get_threshold_condition_code(self):
        return "false" if self.model["threshold"] is None else f"{self.threshold} >= 0"

    @staticmethod
    def from_val_descriptors(model, output_var_name: str,inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys())
        return AutoNeuronModel(model, output_var_name, param_vals, var_vals)
        
class AutoSynapseModel(AutoModel):
    @staticmethod
    def from_val_descriptors(model, str,inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys())
        return AutoSynapseModel(model, param_vals, var_vals)
        