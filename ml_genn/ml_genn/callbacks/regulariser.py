import logging
import numpy as np

from itertools import chain

from pygenn import SynapseMatrixConnectivity, VarAccessDim
from typing import Optional
from .callback import Callback
from ..utils.network import ConnectionType

from pygenn import get_var_access_dim
from ..utils.network import get_underlying_conn
from ..connection import Connection

logger = logging.getLogger(__name__)

class Regulariser(Callback):
    """
    Args:
        conn:               Synapse population to record from        
    """
    def __init__(self, conn: Connection, strength: float):
        # Get underlying connection
        self._conn = get_underlying_conn(conn)
        self._strength = strength
        self._var_weight = "g"
        self._var ="Gradient"
        

    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network

                        

    def on_batch_end(self, batch: int, metrics):
        # Copy variable from device
        conn = self._compiled_network.connection_populations[self._conn]
        conn.vars[self._var].pull_from_device()
        conn.vars[self._var_weight].pull_from_device()
        # Get view, sliced by batch mask if simulation is batched
        var_view = conn.vars[self._var].view
        var_weight_view = np.sign(conn.vars[self._var_weight].view)
        # L1
        var_view[:] += self._strength * var_weight_view
        conn.vars[self._var].push_to_device()
