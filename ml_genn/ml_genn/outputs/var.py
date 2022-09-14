import numpy as np

from .output import Output
from ..utils.model import NeuronModel


class Var(Output):
    def add_output_logic(self, model: NeuronModel):
        self.output_var_name = model.output_var_name

        if "var_name_types" not in model.model:
            raise RuntimeError("Var output can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("Var output requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = (v for v in model.model["var_name_types"]
                          if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not variable "
                               f"{self.output_var_name} to read")

        return model

    def get_output(self, genn_pop, batch_size:int, shape) -> np.ndarray:
        # Pull variable from genn
        genn_pop.pull_var_from_device(self.output_var_name)

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars[self.output_var_name].view,
                          (batch_size,) + shape)
