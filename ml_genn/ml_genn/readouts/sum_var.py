import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class SumVar(Readout):
    """Read out per-neuron sum of neuron model's output variable"""
    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        self.output_var_name = model.output_var_name

        if "vars" not in model.model:
            raise RuntimeError("SumVar readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("SumVar readout requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = next(v for v in model.model["vars"]
                              if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to sum")

        # Make copy of model
        model_copy = deepcopy(model)

        # Determine name and type of sum variable
        sum_var_name = self.output_var_name + "Sum"
        self.output_var_type = output_var[1]

        # Add code to update sum variable
        model_copy.append_sim_code(
            f"{sum_var_name} += {self.output_var_name};")

        # Add sum variable with same type as output
        # variable and initialise to zero
        model_copy.add_var(sum_var_name, self.output_var_type, 0)

        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        sum_var = genn_pop.vars[self.output_var_name + "Sum"]

        # Pull variable from genn
        sum_var.pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(sum_var.view, (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [(self.output_var_name + "Sum", self.output_var_type, 0.0)]
