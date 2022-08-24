import numpy as np

from .output import Output
from ..utils.model import NeuronModel

from copy import deepcopy


class SpikeCount(Output):
    def __call__(self, model: NeuronModel, output_var_name=None):
        # If model isn't spiking, give error
        if "threshold_condition_code" not in model.model:
            raise RuntimeError("SpikeCount output can only "
                               "be used with spiking models")

        # Make copy of model
        model_copy = deepcopy(model)

        # If model doesn't have variables or reset code, add empty
        # **YUCK**
        if "var_name_types" not in model_copy.model:
            model_copy.model["var_name_types"] = []
        if "reset_code" not in model_copy.model:
            model_copy.model["reset_code"] = ""

        # Add code to increment spike count
        model_copy.model["reset_code"] += "\n$(Scount)++;\n"

        # Add integer spike count variable
        model_copy.model["var_name_types"].append(("Scount", "unsigned int"))

        # Initialise to zero
        model_copy.var_vals["Scount"] = 0

        return model_copy

    def get_output(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull spike count from genn
        genn_pop.pull_var_from_device("Scount")

        # Return contents, reshaped as desired
        return np.reshape(genn_pop.vars["Scount"].view,
                          (batch_size,) + shape)
