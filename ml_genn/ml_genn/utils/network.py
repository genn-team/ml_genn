from typing import Sequence, Union
from ..population import Population
from ..layer import InputLayer, Layer

def get_underlying_pop(obj: Union[InputLayer, Layer, Population]):
    # Get underling population from object
    if isinstance(obj, Population):
        return obj
    elif isinstance(obj, (InputLayer, Layer)):
        return obj.population()
    else:
        raise RuntimeError(f"{obj} is not a valid Population, "
                           f"InputLayer or Layer object")

def get_network_dag(inputs, outputs):
    # Convert inputs and outputs to tuples
    inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
    outputs = outputs if isinstance(outputs, Sequence) else (outputs,)
    
    # Construct topologically sorted list of layers using Kahn's algorithm as
    # described here: https://en.wikipedia.org/wiki/Topological_sorting)
    dag = []
    new_populations = set(get_underlying_pop(i) for i in inputs)
    seen_connections = set()
    while new_populations:
        pop = new_populations.pop()
        dag.append(pop)

        # Explore outgoing connections whose 
        # upstream connections have all been seen
        for conn in pop.outgoing_connections:
            seen_connections.add(conn)
            if seen_connections.issuperset(conn().target().incoming_connections):
                new_populations.add(conn().target())
    
    # Check that output layers are in the DAG i.e. reachable from input layers
    if not all(get_underlying_pop(o) in dag
               for o in outputs):
        raise RuntimeError("outputs unreachable from inputs")
    
    return dag
