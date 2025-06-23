from __future__ import annotations

from typing import Sequence
from .serialisers import Serialiser

from .utils.module import get_object
from .utils.value import set_values

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Connection, Population


class Network:
    """Unstructured network model
    
    Attributes:
        populations:    List of all populations in network
        connections:    List of all connections in network
    """
    _context = None

    def __init__(self):
        self.populations = []
        self.connections = []

    @staticmethod
    def _add_population(pop: Population):
        if Network._context is None:
            raise RuntimeError("Population must be created "
                               "inside a ``with network:`` block")
        Network._context.populations.append(pop)

    @staticmethod
    def _add_connection(conn: Connection):
        if Network._context is None:
            raise RuntimeError("Connection must be created "
                               "inside a ``with network:`` block")
        Network._context.connections.append(conn)

    def __enter__(self):
        if Network._context is not None:
            raise RuntimeError("Nested networks are not currently supported")

        Network._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert Network._context is not None
        Network._context = None
