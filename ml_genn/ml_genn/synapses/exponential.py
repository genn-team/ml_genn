import numpy as np

from .synapse import Synapse
from ..utils.model import SynapseModel
from ..utils.value import InitValue, ValueDescriptor

from ..utils.value import is_value_initializer

genn_model = {
    "param_name_types": [("ExpDecay", "scalar")],
    "sim_code":
        """
        injectCurrent(inSyn);
        inSyn *= ExpDecay;
        """}

class Exponential(Synapse):
    tau = ValueDescriptor("ExpDecay", lambda val, dt: np.exp(-dt / val))

    def __init__(self, tau: InitValue = 5.0):
        super(Exponential, self).__init__()

        self.tau = tau

        if is_value_initializer(self.tau):
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")

    def get_model(self, connection, dt, batch_size):
        return SynapseModel.from_val_descriptors(genn_model, self, dt)
