import numpy as np

from .output import Output
from ..utils import NeuronModel, Value

from copy import deepcopy

class Var(Output):
    def __call__(self, model: NeuronModel, output_var_name=None):
        self.output_var_name = self.output_var_name
        
        if not "var_name_types" in model.model:
            raise RuntimeError("Var output can only be used "
                               "with models with state variables")
        if output_var_name is None:
            raise RuntimeError("Var output requires that models "
                               "specify an output variable name")
        
        # Find output variable
        try:
            output_var = (v for v in model.model["var_name_types"]
                          if v[0] == output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not variable "
                               f"{output_var_name} to read")
       
        return model

    def get_output(self, genn_pop, batch_size, shape):
        # Pull variable from genn
        genn_pop.pull_var_from_device(self.output_var_name)
        
        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars[self.output_var_name].view,
                          (batch_size,) + shape)
