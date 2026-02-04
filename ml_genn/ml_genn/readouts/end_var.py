import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class EndVar(Readout):
    """Read out final value of neuron model's output variable"""
    def add_readout_logic(self, model: NeuronModel, **kwargs):
        self.output_var_name = model.output_var_name

        if "vars" not in model.model:
            raise RuntimeError("EndVar readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("EndVar readout requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = next(v for v in model.model["vars"]
                              if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to read out")
        self.output_var_type = output_var[1]

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        end_var = genn_pop.vars[self.output_var_name]

        # Pull variable from genn
        end_var.pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(end_var.view, (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [(self.output_var_name, self.output_var_type, 0.0)]
