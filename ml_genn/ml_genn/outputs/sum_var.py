import numpy as np

from .output import Output
from ..utils.model import NeuronModel

from copy import deepcopy

class SumVar(Output):
    def __call__(self, model: NeuronModel, output_var_name=None):
        self.output_var_name = self.output_var_name
        
        if not "var_name_types" in model.model:
            raise RuntimeError("SumVar output can only be used "
                               "with models with state variables")
        if output_var_name is None:
            raise RuntimeError("SumVar output requires that models "
                               "specify an output variable name")
        
        # Find output variable
        try:
            output_var = (v for v in model.model["var_name_types"]
                          if v[0] == output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not variable "
                               f"{output_var_name} to sum")
       
        # Make copy of model
        model_copy = deepcopy(model)
        
        # If model doesn't have variables or reset code, add empty
        # **YUCK**
        if not "var_name_types" in model_copy.model:
            model_copy.model["var_name_types"] = []
        if not "sim_code" in model_copy.model:
            model_copy.model["sim_code"] = ""

        # Determine name of sum variable
        sum_var_name = output_var_name + "Sum"

        # Add code to update sum variable
        model_copy.model["sim_code"] += f"\n$({sum_var_name}) += $({output_var_name});\n"

        # Add sum variable with same type as output variable
        model_copy.model["var_name_types"].append((sum_var_name, output_var[1]))

        # Initialise to zero
        model_copy.var_vals[sum_var_name] = 0
        
        return model_copy

    def get_output(self, genn_pop, batch_size, shape):
        sum_var_name = self.output_var_name + "Sum"
        
        # Pull variable from genn
        genn_pop.pull_var_from_device(sum_var_name)
        
        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars[sum_var_name].view,
                          (batch_size,) + shape)
