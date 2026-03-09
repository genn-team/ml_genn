import numpy as np

from .readout import TimeWindowReadout
from ..utils.model import NeuronModel

from copy import deepcopy


class AvgVarExpWeight(TimeWindowReadout):
    """Read out per-neuron average of neuron model's output variable
    with exponential weighting as described by [Nowotny2024]_."""

    def add_readout_logic(self, model: NeuronModel, **kwargs):
        self.output_var_name = model.output_var_name

        if "vars" not in model.model:
            raise RuntimeError("AvgVarExpWeight readout can only be used "
                               "with models with state variables")
        if self.output_var_name is None:
            raise RuntimeError("AvgVarExpWeight readout requires that models "
                               "specify an output variable name")

        # Find output variable
        try:
            output_var = next(v for v in model.model["vars"]
                              if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(f"Model does not have variable "
                               f"{self.output_var_name} to average")

        # Determine name and type of average variable
        avg_var_name = self.output_var_name + "Avg"
        self.output_var_type = output_var[1]

        # Add code to update average variable
        window_start, window_end = self.window_start_end(**kwargs)
        scale = kwargs["dt"] / (window_end - window_start)
        local_t_scale = 1.0 / (window_end - window_start)
        model.append_sim_code(
            self.windowed_readout_code(f"{avg_var_name} += exp(-((t-{window_start}) * {local_t_scale})) * {scale} * {self.output_var_name};", **kwargs))
        # Add average variable with same type as output
        # variable and initialise to zero
        model.add_var(avg_var_name, self.output_var_type, 0)

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        avg_var = genn_pop.vars[self.output_var_name + "Avg"]

        # Pull variable from genn
        avg_var.pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(avg_var.view, (batch_size,) + shape)

    @property
    def reset_vars(self):
        return [(self.output_var_name + "Avg", self.output_var_type, 0.0)]
