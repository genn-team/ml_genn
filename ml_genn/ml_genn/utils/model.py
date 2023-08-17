import numpy as np

from numbers import Number
from typing import Any, MutableMapping, Sequence, Union
from pygenn.genn_wrapper.Models import (VarAccess_READ_ONLY,
                                        VarAccess_READ_WRITE,
                                        VarAccessMode_READ_WRITE)
from .value import Value

from copy import deepcopy
from textwrap import dedent
from pygenn.genn_model import init_var
from .value import (get_values, is_value_constant,
                    is_value_array, is_value_initializer)

EGPValue = Union[Number, Sequence[Number], np.ndarray]


class Model:
    def __init__(self, model: MutableMapping[str, Any],
                 param_vals: MutableMapping[str, Value] = {},
                 var_vals: MutableMapping[str, Value] = {},
                 egp_vals: MutableMapping[str, EGPValue] = {}):
        self.model = model

        self.param_vals = param_vals
        self.var_vals = var_vals
        self.egp_vals = egp_vals

    def has_param(self, name):
        return self._is_in_list("param_name_types", name)

    def has_var(self, name):
        return self._is_in_list("var_name_types", name)

    def has_egp(self, name):
        return self._is_in_list("extra_global_params", name)

    def add_param(self, name: str, type: str, value: Value):
        assert not self.has_param(name)
        self._add_to_list("param_name_types", (name, type))
        self.param_vals[name] = value

    def add_var(self, name: str, type: str, value: Value,
                access_mode: int = VarAccess_READ_WRITE):
        assert not self.has_var(name)
        self._add_to_list("var_name_types", (name, type, access_mode))
        self.var_vals[name] = value

    def add_egp(self, name: str, type: str, value: EGPValue):
        assert not self.has_egp(name)
        self._add_to_list("extra_global_params", (name, type))
        self.egp_vals[name] = value
    
    def convert_param_to_egp(self, param_name: str):
        # Search for parameter definition
        param_index, param = self._search_list("param_name_types", param_name)

        # Remove parameter from model
        self.model["param_name_types"].pop(param_index)

        # Remove parameter value from dictionary
        param_val = self.param_vals.pop(param_name)

        # Give error if value is an initialiser
        if is_value_initializer(param_val):
            raise RuntimeError("Parameters with values specified with "
                               "initialisers cannot be converted to EGPs.")

        # Add EGP to replace it
        self.add_egp(param_name, param[1], param_val)

    def set_var_access_mode(self, name: str, access_mode: int):
        self._set_access_model("var_name_types", name, access_mode)

    def make_param_var(self, param_name: str, 
                       access_mode: int = VarAccess_READ_ONLY):
        self._make_param_var("var_name_types", param_name, 
                             self.param_vals, self.var_vals, access_mode)

    def process(self):
        # Make copy of model
        model_copy = deepcopy(self.model)

        # Remove param names and types from copy of model (those that will "
        # be implemented as GeNN parameters will live in param_names)
        if "param_name_types" in model_copy:
            param_name_types = model_copy["param_name_types"]
            del model_copy["param_name_types"]
        else:
            param_name_types = []

        # If there aren't any variables already, add dictionary
        if "var_name_types" not in model_copy:
            model_copy["var_name_types"] = []

        # Convert any initializers to GeNN
        var_vals_copy = {}
        var_egp = {}
        for name, val in self.var_vals.items():
            if is_value_initializer(val):
                snippet = val.get_snippet()
                var_vals_copy[name] = init_var(snippet.snippet,
                                               snippet.param_vals)
                var_egp[name] = snippet.egp_vals
            elif is_value_array(val):
                var_vals_copy[name] = val.flatten()
            else:
                var_vals_copy[name] = val

        # Loop through parameters in model
        model_copy["param_names"] = []
        constant_param_vals = {}
        for name, ptype in param_name_types:
            # Get value
            val = self.param_vals[name]

            # If value is a plain number, add it's name to parameter names
            if is_value_constant(val):
                model_copy["param_names"].append(name)
                constant_param_vals[name] = val
            # Otherwise, turn it into a (read-only) variable
            else:
                model_copy["var_name_types"].append((name, ptype,
                                                     VarAccess_READ_ONLY))
                if is_value_initializer(val):
                    snippet = val.get_snippet()
                    var_vals_copy[name] = init_var(snippet.snippet,
                                                   snippet.param_vals)
                    var_egp[name] = snippet.egp_vals
                elif is_value_array(val):
                    var_vals_copy[name] = val.flatten()
                else:
                    var_vals_copy[name] = val

        # Return modified model and; params, var values and EGPs
        return (model_copy, constant_param_vals, var_vals_copy, 
                self.egp_vals, var_egp)

    @property
    def reset_vars(self):
        return self._get_reset_vars("var_name_types", self.var_vals)

    def _search_list(self, name: str, value: str):
        item = [(i, p) for i, p in enumerate(self.model[name])
                if p[0] == value]
        assert len(item) == 1
        return item[0]
    
    def _add_to_list(self, name: str, value):
        if name not in self.model:
            self.model[name] = []
        self.model[name].append(value)

    def _is_in_list(self, name: str, value):
        if name in self.model:
            try:
                next(p for p in self.model[name] if p[0] == value)
                return True
            except StopIteration:
                pass

        return False

    def _append_code(self, name: str, code: str):
        code = dedent(code)
        if name not in self.model:
            self.model[name] = f"{code}\n"
        else:
            self.model[name] += f"\n{code}\n"

    def _prepend_code(self, name: str, code: str):
        code = dedent(code)
        if name not in self.model:
            self.model[name] = f"{code}\n"
        else:
            self.model[name] = f"{code}\n" + self.model[name]

    def _replace_code(self, name: str, source: str, target: str):
        if name in self.model:
            self.model[name] = self.model[name].replace(source, target)

    def _set_access_model(self, name: str, var: str, access_mode):
        # Find var
        var_index, _ = self._search_list(name, var)

        # Take first two elements of existing var and add access mode
        var_array = self.model[name]
        var_array[var_index] = var_array[var_index][:2] + (access_mode,)

    def _make_param_var(self, var_name: str, param_name: str,
                        param_vals, var_vals, access_mode):
        # Search for parameter definition
        param_index, param = self._search_list("param_name_types", param_name)

        # Remove parameter
        self.model["param_name_types"].pop(param_index)

        # Add variable to replace it
        self._add_to_list(var_name, param[:2] + (access_mode,))

        # Remove parameter value and add into var vals
        var_vals[param_name] = param_vals.pop(param_name)

    def _get_reset_vars(self, name: str, var_vals):
        reset_vars = []
        if name in self.model:
            # Loop through them
            for v in self.model[name]:
                # If variable either has default (read-write)
                # access or this is explicitly set
                # **TODO** mechanism to exclude variables from reset
                if len(v) < 3 or (v[2] & VarAccessMode_READ_WRITE) != 0:
                    reset_vars.append((v[0], v[1], var_vals[v[0]]))
        return reset_vars


class CustomUpdateModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, 
                 var_refs={}, egp_vals={}, egp_refs={}):
        super(CustomUpdateModel, self).__init__(model, param_vals,
                                                var_vals, egp_vals)

        self.var_refs = var_refs
        self.egp_refs = egp_refs

    def has_var_ref(self, name):
        return self._is_in_list("var_refs", name)

    def has_egp_ref(self, name):
        return self._is_in_list("egp_refs", name)

    def add_var_ref(self, name, type, value):
        assert not self.has_var_ref(name)
        self._add_to_list("var_refs", (name, type))
        self.var_refs[name] = value

    def set_var_ref_access_mode(self, name, access_mode):
        self._set_access_model("var_refs", name, access_mode)

    def add_egp_ref(self, name, type, value):
        assert not self.has_egp_ref(name)
        self._add_to_list("egp_refs", (name, type))
        self.egp_refs[name] = value

    def append_update_code(self, code):
        self._append_code("update_code", code)
    
    def process(self):
        return (super(CustomUpdateModel, self).process() 
                + (self.var_refs,) + (self.egp_refs,))


class NeuronModel(Model):
    def __init__(self, model, output_var_name,
                 param_vals={}, var_vals={}, egp_vals={}):
        super(NeuronModel, self).__init__(model, param_vals, 
                                          var_vals, egp_vals)

        self.output_var_name = output_var_name

    def add_additional_input_var(self, name, type, init_val):
        self._add_to_list("additional_input_vars", (name, type, init_val))

    def append_sim_code(self, code):
        self._append_code("sim_code", code)
    
    def prepend_sim_code(self, code):
        self._prepend_code("sim_code", code)

    def append_reset_code(self, code):
        self._append_code("reset_code", code)

    def prepend_reset_code(self, code):
        self._prepend_code("reset_code", code)

    def replace_sim_code(self, source: str, target: str):
        self._replace_code("sim_code", source, target)

    def replace_threshold_condition_code(self, source: str, target: str):
        self._replace_code("threshold_condition_code", source, target)

    def replace_reset_code(self, source: str, target: str):
        self._replace_code("reset_code", source, target)

    @staticmethod
    def from_val_descriptors(model, output_var_name, inst, dt,
                             param_vals={}, var_vals={}, egp_vals={}):
        return NeuronModel(
            model, output_var_name, 
            get_values(inst, model.get("param_name_types", []), dt, param_vals),
            get_values(inst, model.get("var_name_types", []), dt, var_vals),
            egp_vals)

    @property
    def output_var(self):
        # Check model has variables and that
        # output variable name is specified
        if "var_name_types" not in self.model:
            raise RuntimeError("Model has no state variables")
        if self.output_var_name is None:
            raise RuntimeError("Output variable not specified")

        # Find output variable
        _, output_var = self._search_list("var_name_types",
                                          self.output_var_name)
        return output_var

class SynapseModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, egp_vals={}):
        super(SynapseModel, self).__init__(model, param_vals, 
                                           var_vals, egp_vals)
    
    @staticmethod
    def from_val_descriptors(model, inst, dt, 
                             param_vals={}, var_vals={}, egp_vals={}):
        return SynapseModel(
            model, 
            get_values(inst, model.get("param_name_types", []), dt, param_vals),
            get_values(inst, model.get("var_name_types", []), dt, var_vals),
            egp_vals)


class WeightUpdateModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, pre_var_vals={},
                 post_var_vals={}, egp_vals={}):
        super(WeightUpdateModel, self).__init__(model, param_vals, 
                                                var_vals, egp_vals)
        
        self.pre_var_vals = pre_var_vals
        self.post_var_vals = post_var_vals
    
    def append_synapse_dynamics(self, code):
        self._append_code("synapse_dynamics_code", code)
    
    def append_sim_code(self, code):
        self._append_code("sim_code", code)
    
    def append_event_code(self, code):
        self._append_code("event_code", code)

    def process(self):
        return (super(WeightUpdateModel, self).process() 
                + (self.pre_var_vals, self.post_var_vals))

    @property
    def reset_pre_vars(self):
        return self._get_reset_vars("pre_var_name_types", self.pre_var_vals)

    @property
    def reset_post_vars(self):
        return self._get_reset_vars("post_var_name_types", self.post_var_vals)
