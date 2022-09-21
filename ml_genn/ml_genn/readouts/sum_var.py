import numpy as np

from .output import Output
from ..utils.model import NeuronModel

from copy import deepcopy


class SumVar(Output):
    def add_output_logic(self, model: NeuronModel):
        self.output_var_name = model.output_var_name

        if "var_name_types" not in model.model:
            raise RuntimeError("SumVar output can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("SumVar output requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = [v for v in model.model["var_name_types"]
                          if v[0] == self.output_var_name]
        except StopIteration:
            raise RuntimeError(f"Model does not variable "
                               f"{self.output_var_name} to sum")

        # Make copy of model
        model_copy = deepcopy(model)

        # Determine name and type of sum variable
        sum_var_name = self.output_var_name + "Sum"
        self.output_var_type = output_var[0][1]

        # Add code to update sum variable
        model_copy.append_sim_code(
            f"$({sum_var_name}) += $({self.output_var_name});")

        # Add sum variable with same type as output variable and initialise to zero
        model_copy.add_var(sum_var_name, self.output_var_type, 0)

        return model_copy

    def get_output(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        sum_var_name = self.output_var_name + "Sum"

        # Pull variable from genn
        genn_pop.pull_var_from_device(sum_var_name)

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars[sum_var_name].view,
                          (batch_size,) + shape)
    
    @property
    def reset_vars(self):
        return [(self.output_var_name + "Sum", self.output_var_type, 0.0)]