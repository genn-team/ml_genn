from __future__ import annotations

from typing import Union, TYPE_CHECKING
from .synapse import Synapse
from ..utils.auto_model import AutoSynapseModel
from ..utils.model import SynapseModel

if TYPE_CHECKING:
    from .. import Connection

genn_model = {
    "sim_code":
        """
        injectCurrent(inSyn);
        inSyn = 0;
        """}
        
class Delta(Synapse):
    """Synapse model where inputs produce instantaneous
    voltage jumps in target neurons."""
    def __init__(self):
        super(Delta, self).__init__()

    def get_model(self, connection: Connection, dt: float,
                  batch_size: int) -> Union[AutoSynapseModel, SynapseModel]:
        # Build basic model
        genn_model = {"vars": {"I": (None, "I + weight")}}
        
        return AutoSynapseModel.from_val_descriptors(genn_model, self)
