import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class AvgVar(Readout):
    """Read out per-neuron average of neuron model's output variable"""
    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        self.output_var_name = model.output_var_name

        if "vars" not in model.model:
            raise RuntimeError("AvgVar readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("AvgVar readout requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = next(v for v in model.model["vars"]
                              if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to average")

        # Make copy of model
        model_copy = deepcopy(model)

        # Determine name and type of average variable
        avg_var_name = self.output_var_name + "Avg"
        self.output_var_type = output_var[1]

        # Add code to update average variable
        scale = 1.0 / kwargs["example_timesteps"]
        model_copy.append_sim_code(
            f"{avg_var_name} += {scale} * {self.output_var_name};")

        # Add average variable with same type as output
        # variable and initialise to zero
        model_copy.add_var(avg_var_name, self.output_var_type, 0)

        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        avg_var = genn_pop.vars[self.output_var_name + "Avg"]

        # Pull variable from genn
        avg_var.pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(avg_var.view, (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [(self.output_var_name + "Avg", self.output_var_type, 0.0)]
