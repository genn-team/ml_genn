import numpy as np
import sympy

from typing import Any, MutableMapping, Optional, Tuple
from .model import NeuronModel, SynapseModel
from .value import Value

from copy import deepcopy
from itertools import chain
from ..utils.value import get_auto_values

Variables = MutableMapping[str, Tuple[Optional[str], Optional[str]]]

class AutoModel:
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: Optional[MutableMapping[str, Value]] = None,
                 var_vals: Optional[MutableMapping[str, Value]] = None):
        self.model = model

        self.param_vals = param_vals or {}
        self.var_vals = var_vals or {}

        # Create sympy symbols for variable and parameter names
        # **NOTE** model doesn't explicitly list parameters
        self.symbols = {n: sympy.Symbol(n) 
                        for n in chain(self.var_vals.keys(),
                                       self.param_vals.keys())}
        
        # If model has any variables
        if "vars" in self.model:
            # Parse ODEs
            self.dx_dt =\
                {sympy.Symbol(n): sympy.parse_expr(v[0], 
                                                   local_dict=self.symbols)
                 for n, v in self.model["vars"].items()
                 if v[0] is not None}
            
            # Parse jumps
            self.jumps =\
                {sympy.Symbol(n): sympy.parse_expr(v[1], 
                                                   local_dict=self.symbols)
                 for n, v in self.model["vars"].items()
                 if v[1] is not None}
        else:
            self.dx_dt = {}


    def get_vars(self, var_type: str = "scalar"):
        return [(n, var_type) for n in self.var_vals.keys()]
    
    def get_params(self, param_type: str = "scalar"):
        return [(n, param_type) for n in self.param_vals.keys()]
        
class AutoNeuronModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any], output_var_name: str,
                 param_vals: Optional[MutableMapping[str, Value]] = None,
                 var_vals: Optional[MutableMapping[str, Value]] = None):
        super(AutoNeuronModel, self).__init__(model, param_vals, var_vals)

        self.output_var_name = output_var_name
        
        if "threshold" in self.model and self.model["threshold"] is not None:
            self.threshold = sympy.parse_expr(self.model["threshold"],
                                              local_dict=self.symbols)
        else:
            self.threshold = 0

    # **TODO** property
    def get_threshold_condition_code(self):
        return (f"({sympy.ccode(self.threshold)}) >= 0" 
                if self.threshold != 0
                else "")

    # **TODO** property
    def get_reset_code(self):
        jumps = [f"{n} = {v[1]};" for n, v in self.model["vars"].items()
                 if v[1] is not None and v[1] != n]
        return "\n".join(jumps)
    
    @staticmethod
    def from_val_descriptors(model, output_var_name: str,inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys(),
                                               param_vals, var_vals)
        return AutoNeuronModel(model, output_var_name, param_vals, var_vals)
        
class AutoSynapseModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: Optional[MutableMapping[str, Value]] = None,
                 var_vals: Optional[MutableMapping[str, Value]] = None):
        super(AutoSynapseModel, self).__init__(model, param_vals, var_vals)

        if "inject_current" in self.model:
            self.inject_current = sympy.parse_expr(
                self.model["inject_current"], local_dict=self.symbols)
        else:
            raise RuntimeError("AutoSynapseModel requires an "
                               "'inject_current' expression.")

        try:
            # Find first variable that can be implemented using built-in inSyn
            weight_sym = sympy.Symbol("weight")
            rep_sym = next(sym for sym, expr in self.jumps.items()
                           if (expr == (sym + weight_sym)
                               and self.var_vals[sym.name] == 0))
        except StopIteration:
            pass
        finally:
            # Rename variable in dx_dt and jumps as inSyn
            # and substitute in all jumps and dynamics
            in_syn_sym = sympy.Symbol("inSyn")
            self.dx_dt = {in_syn_sym if sym == rep_sym else sym: expr.subs(rep_sym, in_syn_sym)
                          for sym, expr in self.dx_dt.items()}
            self.jumps = {in_syn_sym if sym == rep_sym else sym: expr.subs(rep_sym, in_syn_sym)
                          for sym, expr in self.jumps.items()}

            # Also substitue in inject expression            
            self.inject_current = self.inject_current.subs(rep_sym, in_syn_sym)

            # Remove from var_vals 
            # **NOTE** this prevents it getting implemented as a variable
            self.var_vals.pop(rep_sym.name)

    # **TODO** property
    def get_jump_code(self):
        # Generate C code for forward jumps, 
        in_syn_sym = sympy.Symbol("inSyn")
        w_sym = sympy.Symbol("weight")

        # Add jumps for non-inSyn symbols
        clines = [f"{sym.name} += {sympy.ccode(expr.subs(w_sym, in_syn_sym) - sym)};"
                  for sym, expr in self.jumps.items()
                  if sym != in_syn_sym]

        # If an inSyn jump isn't specified, zero it
        if in_syn_sym not in self.jumps:
            clines.append("inSyn = 0;")
        return "\n".join(clines)

    # **TODO** property
    def get_inject_current_code(self):
        return sympy.ccode(self.inject_current)

    @staticmethod
    def from_val_descriptors(model, inst, 
                             param_vals={}, var_vals={}):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys(),
                                               param_vals, var_vals)
        return AutoSynapseModel(model, param_vals, var_vals)
        