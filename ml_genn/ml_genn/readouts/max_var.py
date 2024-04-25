import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class MaxVar(Readout):
    """Read out per-neuron maximum value of neuron model's output variable"""
    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        self.output_var_name = model.output_var_name

        if "vars" not in model.model:
            raise RuntimeError("MaxVar readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("MaxVar readout requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = next(v for v in model.model["vars"]
                              if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to max")

        # Make copy of model
        model_copy = deepcopy(model)

        # Determine name and type of sum variable
        max_var_name = self.output_var_name + "Max"
        self.output_var_type = output_var[1]
        
        # Add max variable with same type as output
        # variable and initialise to zero
        # **TODO** min value of output_var_type
        model_copy.add_var(max_var_name, self.output_var_type, 0)

        # If compiler needs time max occurred at
        if kwargs.get("max_time_required", False):
            # Add variable to hold max time
            # **TODO** should use time type from GeNNModel
            max_time_var_name = self.output_var_name + "MaxTime"
            model_copy.add_var(max_time_var_name, "scalar", 0)
            
            # Add code to update max variable and time 
            model_copy.append_sim_code(
                f"""
                if ($({self.output_var_name}) > $({max_var_name})) {{
                    $({max_var_name})= $({self.output_var_name});
                    $({max_time_var_name}) = t;
                }}
                """)
        # Otherwise, just add code to update max variable
        else:
            model_copy.append_sim_code(
                f"""
                if ($({self.output_var_name}) > $({max_var_name})) {{
                    $({max_var_name})= $({self.output_var_name});
                }}
                """)

        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        max_var = genn_pop.vars[self.output_var_name + "Max"]

        # Pull variable from genn
        max_var.pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(max_var.view, (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [(self.output_var_name + "Max", self.output_var_type, 0.0)]