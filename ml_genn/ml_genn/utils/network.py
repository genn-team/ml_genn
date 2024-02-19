from typing import Sequence, Union
from ..connection import Connection
from ..population import Population
from ..layer import InputLayer, Layer

ConnectionType = Union[Connection, Layer]
PopulationType = Union[InputLayer, Layer, Population]
ConnectionType = Union[Layer, Connection]

def get_underlying_pop(obj: PopulationType) -> Population:
    # Get underling population from object
    if isinstance(obj, Population):
        return obj
    elif isinstance(obj, (InputLayer, Layer)):
        return obj.population()
    else:
        raise RuntimeError(f"{obj} is not a valid Population, "
                           f"InputLayer or Layer object")

def get_underlying_conn(obj: ConnectionType) -> Connection:
    # Get underling connection from object
    if isinstance(obj, Connection):
        return obj
    elif isinstance(obj, Layer):
        return obj.connection()
    else:
        raise RuntimeError(f"{obj} is not a valid Connection or Layer object")

def get_underlying_conn(obj: PopulationType) -> Connection:
    # Get underling connection from object
    if isinstance(obj, Connection):
        return obj
    elif isinstance(obj, Layer) and obj.connection() is not None:
        return obj.connection()
    else:
        raise RuntimeError(f"{obj} is not a valid Connection object "
                           f"or Layer with associated connection")


def get_network_dag(inputs, outputs):
    # Convert inputs and outputs to tuples
    inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
    outputs = outputs if isinstance(outputs, Sequence) else (outputs,)

    # Construct topologically sorted list of layers using Kahn's algorithm as
    # described here: https://en.wikipedia.org/wiki/Topological_sorting)
    dag = []
    new_pops = set(get_underlying_pop(i) for i in inputs)
    seen_conns = set()
    while new_pops:
        pop = new_pops.pop()
        dag.append(pop)

        # Explore outgoing connections whose
        # upstream connections have all been seen
        for conn in pop.outgoing_connections:
            seen_conns.add(conn)
            if seen_conns.issuperset(conn().target().incoming_connections):
                new_pops.add(conn().target())

    # Check that output layers are in the DAG i.e. reachable from input layers
    if not all(get_underlying_pop(o) in dag
               for o in outputs):
        raise RuntimeError("outputs unreachable from inputs")

    return dag
