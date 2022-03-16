from .output import Output
from ..utils import Value

from copy import deepcopy

class SpikeCount(Output):
    def __call__(self, model, output_var_name=None):
        # If model isn't spiking, give error
        if not "threshold_condition_code" in model.model:
            raise RuntimeError("SpikeCount output can only "
                               "be used with spiking models")

        # Make copy of model
        model_copy = deepcopy(model)

        # If model doesn't have variables or reset code, add empty
        # **YUCK**
        if not "var_name_types" in model_copy.model:
            model_copy.model["var_name_types"] = []
        if not "reset_code" in model_copy.model:
            model_copy.model["reset_code"] = ""

        # Add code to increment spike count
        model_copy.model["reset_code"] += "\n$(Scount)++;\n"

        # Add integer spike count variable
        model_copy.model["var_name_types"].append(("Scount", "unsigned int"))
        
        # Initialise to zero
        model_copy.var_vals["Scount"] = Value(0)
        
        return model_copy