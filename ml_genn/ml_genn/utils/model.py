class Model:
    def __init__(self, model, param_vals={}, var_vals={}, egp_vals={}):
        self.model = model
    
        self.param_vals = param_vals
        self.var_vals = var_vals
        self.egp_vals = egp_vals
    
    def add_param(self, name, type, value):
        self._add_to_list("param_name_types", (name, type))
        self.param_vals[name] = value

    def add_var(self, name, type, value):
        self._add_to_list("var_name_types", (name, type))
        self.var_vals[name] = value

    def add_egp(self, name, type, value):
        self._add_to_list("extra_global_params", (name, type))
        self.egp_vals[name] = value

    def _add_to_list(self, name, value):
        if name not in self.model:
            self.model[name] = []
        self.model[name].append(value)

    def _append_code(self, name, code):
        if name not in self.model:
            self.model[name] = f"{code}\n"
        else:
            self.model[name] += f"\n{code}\n"

class CustomUpdateModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, 
                 var_refs={}, egp_vals={}):
        super(CustomUpdateModel, self).__init__(model, param_vals,
                                                var_vals, egp_vals)
        
        self.var_refs = var_refs

    def add_var_ref(self, name, type, value):
        self._add_to_list("var_refs", (name, type))
        self.var_refs[name] = value
    
    def append_update_code(self, code):
        self._append_code("update_code", code)

class NeuronModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, egp_vals={}):
        super(NeuronModel, self).__init__(model, param_vals, 
                                          var_vals, egp_vals)
    
    def add_additional_input_var(self, name, type, init_val):
        self._add_to_list("additional_input_vars", (name, type, init_val))

    def append_sim_code(self, code):
        self._append_code("sim_code", code)

    def append_reset_code(self, code):
        self._append_code("reset_code", code)

class SynapseModel(Model):
    def __init__(self, model, param_vals={}, var_vals={}, egp_vals={}):
        super(SynapseModel, self).__init__(model, param_vals, 
                                           var_vals, egp_vals)

