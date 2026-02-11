import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class AvgVarExpWeight(Readout):
    """Read out per-neuron average of neuron model's output variable
    with exponential weighting as described by [Nowotny2024]_."""
    def __init__(self, **kwargs):
        """Through kwargs, allow to define a window in which to average
        the output var with exponential weighting across the window
        from 1 to 1/e. If no window is defined, default to the original
        averaging and exponential weight across the whole trial."""
        self.window_start = kwargs.get("window_start")
        self.window_end = kwargs.get("window_end")

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
        window_start = 0 if self.window_start is None else self.window_start
        window_end = kwargs["example_timesteps"] if self.window_end is None else self.window_end
        scale = kwargs["dt"] / (window_end - window_start)
        local_t_scale = 1.0 / (window_end - window_start)
        model.append_sim_code(
            f"if (t >= {window_start} && t < {window_end}) {avg_var_name} += exp(-((t-{window_start}) * {local_t_scale})) * {scale} * {self.output_var_name};")

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
