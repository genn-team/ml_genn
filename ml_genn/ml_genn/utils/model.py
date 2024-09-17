import numpy as np

from numbers import Number
from typing import Any, MutableMapping, Sequence, Union
from pygenn import VarAccess, VarAccessModeAttribute
from .value import Value

from copy import deepcopy
from textwrap import dedent
from pygenn import init_var
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
        self.dynamic_param_names = set()
        self.var_vals = var_vals
        self.egp_vals = egp_vals

    def has_param(self, name):
        return self._is_in_list("params", name)

    def has_var(self, name):
        return self._is_in_list("vars", name)

    def has_egp(self, name):
        return self._is_in_list("extra_global_params", name)

    def add_param(self, name: str, type: str, value: Value):
        assert not self.has_param(name)
        self._add_to_list("params", (name, type))
        self.param_vals[name] = value

    def add_var(self, name: str, type: str, value: Value,
                access_mode: int = VarAccess.READ_WRITE):
        assert not self.has_var(name)
        self._add_to_list("vars", (name, type, access_mode))
        self.var_vals[name] = value

    def add_egp(self, name: str, type: str, value: EGPValue):
        assert not self.has_egp(name)
        self._add_to_list("extra_global_params", (name, type))
        self.egp_vals[name] = value

    def set_param_dynamic(self, name: str, dynamic: bool = True):
        assert self.has_param(name)
        if dynamic:
            self.dynamic_param_names.add(name)
        else:
            self.dynamic_param_names.discard(name)

    def set_var_access_mode(self, name: str, access_mode: int):
        self._set_access_model("vars", name, access_mode)

    def make_param_var(self, param_name: str, 
                       access_mode: int = VarAccess.READ_ONLY):
        assert param_name not in self.dynamic_param_names
        self._make_param_var("vars", param_name, 
                             self.param_vals, self.var_vals, access_mode)

    def process(self):
        # Make copy of model
        model_copy = deepcopy(self.model)

        # Remove param names and types from copy of model (those that will "
        # be implemented as GeNN parameters will live in params)
        if "params" in model_copy:
            params = model_copy["params"]
            del model_copy["params"]
        else:
            params = []

        # If there aren't any variables already, add dictionary
        if "vars" not in model_copy:
            model_copy["vars"] = []

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
        model_copy["params"] = []
        constant_param_vals = {}
        for name, ptype in params:
            # Get value
            val = self.param_vals[name]

            # If value is a plain number, add to parameters
            if is_value_constant(val):
                model_copy["params"].append((name, ptype))
                constant_param_vals[name] = val
            # Otherwise, turn it into a (read-only) variable
            else:
                assert name not in self.dynamic_param_names
    
                model_copy["vars"].append((name, ptype,
                                           VarAccess.READ_ONLY))
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
        return (model_copy, constant_param_vals, self.dynamic_param_names,
                var_vals_copy, self.egp_vals, var_egp)

    @property
    def reset_vars(self):
        return self._get_reset_vars("vars", self.var_vals)

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
        param_index, param = self._search_list("params", param_name)

        # Remove parameter
        self.model["params"].pop(param_index)

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
                if len(v) < 3 or (v[2] & VarAccessModeAttribute.READ_WRITE) != 0:
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


class CustomConnectivityUpdateModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, pre_var_vals={},
                 post_var_vals={}, var_refs={}, pre_var_refs={}, 
                 post_var_refs={}, egp_vals={}, egp_refs={}):
        super(CustomConnectivityUpdateModel, self).__init__(model, param_vals,
                                                            var_vals, 
                                                            egp_vals)

        self.pre_var_vals = pre_var_vals
        self.post_var_vals = post_var_vals
        self.var_refs = var_refs
        self.pre_var_refs = pre_var_refs
        self.post_var_refs = post_var_refs
        self.egp_refs = egp_refs

    def process(self):
        return (super(CustomConnectivityUpdateModel, self).process() 
                + (self.pre_var_vals,) + (self.post_var_vals,)
                + (self.var_refs,) + (self.pre_var_refs,) 
                + (self.post_var_refs,) + (self.egp_refs,))

                
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
            get_values(inst, model.get("params", []), dt, param_vals),
            get_values(inst, model.get("vars", []), dt, var_vals),
            egp_vals)

    @property
    def output_var(self):
        # Check model has variables and that
        # output variable name is specified
        if "vars" not in self.model:
            raise RuntimeError("Model has no state variables")
        if self.output_var_name is None:
            raise RuntimeError("Output variable not specified")

        # Find output variable
        _, output_var = self._search_list("vars",
                                          self.output_var_name)
        return output_var

class SynapseModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, egp_vals={},
                 neuron_var_refs={}):
        super(SynapseModel, self).__init__(model, param_vals, 
                                           var_vals, egp_vals)

        self.neuron_var_refs = neuron_var_refs

    def process(self):
        return (super(SynapseModel, self).process() 
                + (self.neuron_var_refs,))

    @staticmethod
    def from_val_descriptors(model, inst, dt, 
                             param_vals={}, var_vals={},
                             egp_vals={}, neuron_var_refs={}):
        return SynapseModel(
            model, 
            get_values(inst, model.get("params", []), dt, param_vals),
            get_values(inst, model.get("vars", []), dt, var_vals),
            egp_vals, neuron_var_refs)


class WeightUpdateModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, pre_var_vals={},
                 post_var_vals={}, egp_vals={}, pre_neuron_var_refs={},
                 post_neuron_var_refs={},):
        super(WeightUpdateModel, self).__init__(model, param_vals, 
                                                var_vals, egp_vals)

        self.pre_var_vals = pre_var_vals
        self.post_var_vals = post_var_vals
        self.pre_neuron_var_refs = pre_neuron_var_refs
        self.post_neuron_var_refs = post_neuron_var_refs
    
    def add_pre_neuron_var_ref(self, name, type, target):
        self._add_to_list("pre_neuron_var_refs", (name, type))
        self.pre_neuron_var_refs[name] = target
    
    def add_post_neuron_var_ref(self, name, type, target):
        self._add_to_list("post_neuron_var_refs", (name, type))
        self.post_neuron_var_refs[name] = target
        
    def append_synapse_dynamics(self, code):
        self._append_code("synapse_dynamics_code", code)
    
    def append_pre_spike_syn_code(self, code):
        self._append_code("pre_spike_syn_code", code)
    
    def append_pre_event_syn_code(self, code):
        self._append_code("pre_event_syn_code", code)

    def process(self):
        return (super(WeightUpdateModel, self).process() 
                + (self.pre_var_vals, self.post_var_vals,
                   self.pre_neuron_var_refs, self.post_neuron_var_refs))

    @property
    def reset_pre_vars(self):
        return self._get_reset_vars("pre_vars", self.pre_var_vals)

    @property
    def reset_post_vars(self):
        return self._get_reset_vars("post_vars", self.post_var_vals)
