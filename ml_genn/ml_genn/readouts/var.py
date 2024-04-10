import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel


class Var(Readout):
    """Read out instantaneous value of neuron model's output variable"""
    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        self.output_var_name = model.output_var_name

        if "vars" not in model.model:
            raise RuntimeError("Var readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("Var readout requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            _ = next(v for v in model.model["vars"]
                     if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to read")

        return model

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull variable from genn
        genn_pop.vars[self.output_var_name].pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars[self.output_var_name].view,
                          (batch_size,) + shape)
