from .network import Network


class SequentialNetwork(Network):
    _context = None

    def __init__(self):
        super(SequentialNetwork, self).__init__()

        self.layers = []

    @staticmethod
    def _add_input_layer(layer, population):
        if SequentialNetwork._context is None:
            raise RuntimeError("InputLayer must be created "
                               "inside a ``with sequential_network:`` block")
        SequentialNetwork._context.layers.append(layer)
        SequentialNetwork._context.populations.append(population)

    @staticmethod
    def _add_layer(layer, population, connection):
        if SequentialNetwork._context is None:
            raise RuntimeError("Layer must be created "
                               "inside a ``with sequential_network:`` block")
        SequentialNetwork._context.layers.append(layer)
        SequentialNetwork._context.populations.append(population)
        SequentialNetwork._context.connections.append(connection)

    @staticmethod
    def get_prev_layer():
        if SequentialNetwork._context is None:
            raise RuntimeError("Cannot get previous layer outside of a "
                               "``with sequential_network:`` block")
        if len(SequentialNetwork._context.layers) > 0:
            return SequentialNetwork._context.layers[-1]
        else:
            return None

    def __enter__(self):
        if SequentialNetwork._context is not None:
            raise RuntimeError("Nested sequential networks are "
                               "not currently supported")

        SequentialNetwork._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert SequentialNetwork._context is not None
        SequentialNetwork._context = None
