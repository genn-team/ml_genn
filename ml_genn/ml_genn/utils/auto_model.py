import numpy as np
import sympy

from typing import Any, MutableMapping
from .model import NeuronModel, SynapseModel
from .value import Value

from copy import deepcopy
from itertools import chain
from ..utils.value import get_auto_values

class AutoModel:
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: MutableMapping[str, Value] = {},
                 var_vals: MutableMapping[str, Value] = {}):
        self.model = model

        self.param_vals = param_vals
        self.var_vals = var_vals

        # Create sympy symbols for variable and parameter names
        # **NOTE** model doesn't explicitly list parameters
        self.symbols = {n: sympy.Symbol(n) 
                        for n in chain(self.var_vals.keys(),
                                       self.param_vals.keys())}
        
        # Parse ODEs
        if "vars" in self.model:
            self.dx_dt =\
                {sympy.Symbol(n): sympy.parse_expr(v[0], 
                                                   local_dict=self.symbols)
                 for n, v in self.model["vars"].items()
                 if v[0] is not None}
        else:
            self.dx_dt = {}


    def get_vars(self, var_type: str = "scalar"):
        return [(n, var_type) for n in self.var_vals.keys()]
    
    def get_params(self, param_type: str = "scalar"):
        return [(n, param_type) for n in self.param_vals.keys()]
        
class AutoNeuronModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any], output_var_name: str,
                 param_vals: MutableMapping[str, Value] = {},
                 var_vals: MutableMapping[str, Value] = {}):
        super(AutoNeuronModel, self).__init__(model, param_vals, var_vals)

        self.output_var_name = output_var_name

    def get_threshold_condition_code(self):
        return (f"({self.model['threshold']}) >= 0" if "threshold" in self.model
                else "")

    def get_reset_code(self):
        jumps = [f"{n} = {v[1]};" for n, v in self.model["vars"].items()
                 if v[1] is not None and v[1] != n]
        return "\n".join(jumps)
    
    @staticmethod
    def from_val_descriptors(model, output_var_name: str,inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys())
        return AutoNeuronModel(model, output_var_name, param_vals, var_vals)
        
class AutoSynapseModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: MutableMapping[str, Value] = {},
                 var_vals: MutableMapping[str, Value] = {}):
        super(AutoSynapseModel, self).__init__(model, param_vals, var_vals)

        if "I" not in self.var_vals:
            self.var_vals["I"] = 0.0

    def get_jump_code(self):
        h = {}
        for n, v in self.model["vars"].items():
            if v[1] is not None:
                expr = (sympy.parse_expr(v[1], local_dict=self.symbols) 
                        - self.symbols[n])
                if sympy.diff(expr, self.symbols[n]) == 0:
                    if expr != 0:
                        h[n] = expr
                else:
                    raise NotImplementedError(
                        "EventProp compiler only supports "
                        "synapses which (only) add input to target variables.")
        
        # Generate C code for forward jumps, 
        in_syn_sym = sympy.Symbol("inSyn")
        w_sym = sympy.Symbol("weight")
        clines = [f"{v} += {sympy.ccode(e.subs(w_sym, in_syn_sym))};"
                  for v, e in h.items()]

        clines.append("inSyn = 0;")
        return "\n".join(clines)

    @staticmethod
    def from_val_descriptors(model, inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys())
        return AutoSynapseModel(model, param_vals, var_vals)
        