from typing import Sequence, Union

class Network:
    _context = None

    def __init__(self):
        self.populations = []
        self.connections = []
    
    def get_population_dag(self, inputs, outputs):
        # Convert inputs and outputs to tuples
        inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
        outputs = outputs if isinstance(outputs, Sequence) else (outputs,)
        
        # Construct topologically sorted list of layers (Kahn's algorithm as described here: https://en.wikipedia.org/wiki/Topological_sorting)
        dag = []
        new_populations = set(inputs)
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
        
        print(dag)
        # Check that output layers are reachable from input layers
        #if not all(output in self.layers for output in self.outputs):
        #    raise ValueError('output layers unreachable from input layers')

    @staticmethod
    def add_population(pop):
        if Network._context is None:
            raise RuntimeError("Population must be created "
                               "inside a ``with network:`` block")
        Network._context.populations.append(pop)

    @staticmethod
    def add_connection(conn):
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
