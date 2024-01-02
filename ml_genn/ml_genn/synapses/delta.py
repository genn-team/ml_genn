from .synapse import Synapse
from ..utils.model import SynapseModel

genn_model = {
    "sim_code":
        """
        injectCurrent(inSyn);
        inSyn = 0;
        """}
        
class Delta(Synapse):
    def __init__(self):
        super(Delta, self).__init__()

    def get_model(self, connection, dt, batch_size):
        return SynapseModel(genn_model, {}, {})
