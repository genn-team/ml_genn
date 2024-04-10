from __future__ import annotations
from abc import ABC

from typing import TYPE_CHECKING
from ..utils.model import SynapseModel

if TYPE_CHECKING:
    from .. import Connection

from abc import abstractmethod


class Synapse(ABC):
    """Base class for all synapse models"""

    @abstractmethod
    def get_model(self, connection: Connection,
                  dt: float, batch_size: int) -> SynapseModel:
        """Gets PyGeNN implementation of synapse

        Args:
            connection: Connection this synapse belongs to
            dt :        Timestep of simulation (in ms)
            batch_size: Batch size of the model
        """
        pass
