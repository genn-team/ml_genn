from .network import Network


class SequentialNetwork(Network):
    def __init__(self, default_params: dict = {}):
        super(SequentialNetwork, self).__init__(default_params)

        self.layers = []

    @staticmethod
    def _add_input_layer(layer, population):
        if Network._context is None:
            raise RuntimeError("InputLayer must be created "
                               "inside a ``with sequential_network:`` block")
        Network._context.layers.append(layer)
        Network._context.populations.append(population)

    @staticmethod
    def _add_layer(layer, population, connection):
        if Network._context is None:
            raise RuntimeError("Layer must be created "
                               "inside a ``with sequential_network:`` block")
        Network._context.layers.append(layer)
        Network._context.populations.append(population)
        Network._context.connections.append(connection)

    @staticmethod
    def get_prev_layer():
        if Network._context is None:
            raise RuntimeError("Cannot get previous layer outside of a "
                               "``with sequential_network:`` block")
        if len(Network._context.layers) > 0:
            return Network._context.layers[-1]
        else:
            return None

    def __enter__(self):
        if Network._context is not None:
            raise RuntimeError("Nested networks are "
                               "not currently supported")

        Network._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert Network._context is not None
        Network._context = None
