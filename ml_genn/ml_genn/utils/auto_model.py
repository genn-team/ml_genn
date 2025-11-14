import numpy as np
import sympy

from collections import namedtuple
from collections.abc import Mapping
from typing import Any, MutableMapping, Optional, Tuple
from .value import Value

from ..utils.value import get_auto_values

Variables = MutableMapping[str, Tuple[Optional[str], Optional[str]]]

class AutoModel:
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: Optional[MutableMapping[str, Value]] = None,
                 var_vals: Optional[MutableMapping[str, Value]] = None,
                 solver: str = "exponential_euler",
                 sub_steps: int = 1):
        self.param_vals = param_vals or {}
        self.var_vals = var_vals or {}
        self.solver = solver
        self.sub_steps = sub_steps

        # If model has any variables
        if "vars" in model:
            # Vars can be specified as either dictionaries or tuples,
            # Handle this and correct defaults be converting to a namedtuple
            Var = namedtuple("Var", ["dynamics", "jump", "reset"],
                             defaults=[None, None, True])
            vars = {n: (Var(**v) if isinstance(v, MutableMapping)
                        else Var(*v))
                    for n, v in model["vars"].items()}

            # Parse ODEs
            self.dx_dt =\
                {sympy.Symbol(n): sympy.parse_expr(v[0])
                 for n, v in vars.items()
                 if v.dynamics is not None}
            
            # Parse jumps
            self.jumps =\
                {sympy.Symbol(n): sympy.parse_expr(v[1])
                 for n, v in vars.items()
                 if v.jump is not None}
            
            # Build set of variables which don't get reset
            self.non_reset_vars = set(n for n, v in vars.items()
                                      if not v.reset)
        else:
            self.dx_dt = {}
            self.jumps = {}
            self.non_reset_vars = set()

    def add_var(self, name: str, dynamics: Optional[str], jump: Optional[str],
                value: Value = 0.0, reset: bool = True):
        sym = sympy.Symbol(name)
        if sym in self.dx_dt or sym in self.var_vals or sym in self.jumps:
            raise RuntimeError(f"AutoModel has existing variable: {name}")
        
        # If provided, parse dynamics and jump and add to dicts
        if dynamics is not None:
            self.dx_dt[sym] = sympy.parse_expr(dynamics)
        if jump is not None:
            self.jumps[sym] = sympy.parse_expr(jump)

        # Add value to dictionary
        self.var_vals[name] = value

        # If variable doesn't get reset, add to set
        if not reset:
            self.non_reset_vars.add(name)
    
    def get_vars(self, var_type: str = "scalar"):
        return [(n, var_type) for n in self.var_vals.keys()]
    
    def get_params(self, param_type: str = "scalar"):
        return [(n, param_type) for n in self.param_vals.keys()]
        
class AutoNeuronModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any], output_var_name: str,
                 param_vals: Optional[MutableMapping[str, Value]] = None,
                 var_vals: Optional[MutableMapping[str, Value]] = None,
                 solver: str = "exponential_euler",
                 sub_steps: int = 1):
        super(AutoNeuronModel, self).__init__(model, param_vals, var_vals, solver, sub_steps)

        self.output_var_name = output_var_name
        
        if "threshold" in model and model["threshold"] is not None:
            self.threshold = sympy.parse_expr(model["threshold"])
        else:
            self.threshold = 0

    @property
    def threshold_condition_code(self):
        return (f"({sympy.ccode(self.threshold)}) >= 0" 
                if self.threshold != 0
                else "")

    @property
    def reset_code(self):
        jumps = [f"{v.name} = {sympy.ccode(j)};" 
                 for v, j in self.jumps.items()
                 if j != v]
        return "\n".join(jumps)
    
    @staticmethod
    def from_val_descriptors(model, output_var_name: str,inst, 
                             param_vals={}, var_vals={},
                             solver: str = "exponential_euler",
                             sub_steps: int = 1):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys(),
                                               param_vals, var_vals)
        return AutoNeuronModel(model, output_var_name, param_vals, var_vals, solver, sub_steps)
        
class AutoSynapseModel(AutoModel):
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: Optional[MutableMapping[str, Value]] = None,
                 var_vals: Optional[MutableMapping[str, Value]] = None,
                 solver: str = "exponential_euler",
                 sub_steps: int = 1):
        super(AutoSynapseModel, self).__init__(model, param_vals, var_vals, solver, sub_steps)

        if "inject_current" in model:
            self.inject_current = sympy.parse_expr(model["inject_current"])
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
            # **YUCK** we don't just pop because self.var_vals might well be
            # a reference to a dictionary owned by something else
            self.var_vals = {n: v for n, v in self.var_vals.items()
                             if n != rep_sym.name}

    @property
    def jump_code(self):
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

    @property
    def inject_current_code(self):
        return sympy.ccode(self.inject_current)

    @staticmethod
    def from_val_descriptors(model, inst, 
                             param_vals={}, var_vals={},
                             solver: str = "exponential_euler",
                             sub_steps: int = 1):
        param_vals, var_vals = get_auto_values(inst, 
                                               model.get("vars", {}).keys(),
                                               param_vals, var_vals)
        return AutoSynapseModel(model, param_vals, var_vals, solver, sub_steps)
