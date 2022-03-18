from .model import Model

class SequentialModel(Model):
    _context = None

    def __init__(self):
        super(SequentialModel, self).__init__()

        self.layers = []

    @staticmethod
    def add_input_layer(layer, population):
        if SequentialModel._context is None:
            raise RuntimeError("InputLayer must be created "
                               "inside a ``with sequential_model:`` block")
        SequentialModel._context.layers.append(layer)
        SequentialModel._context.populations.append(population)

    @staticmethod
    def add_layer(layer, population, connection):
        if SequentialModel._context is None:
            raise RuntimeError("Layer must be created "
                               "inside a ``with sequential_model:`` block")
        SequentialModel._context.layers.append(layer)
        SequentialModel._context.populations.append(population)
        SequentialModel._context.connections.append(connection)
    
    @staticmethod
    def get_prev_layer():
        if SequentialModel._context is None:
            raise RuntimeError("Cannot get previous layer outside of a "
                               "``with sequential_model:`` block")
        if len(SequentialModel._context.layers) > 0:
            return SequentialModel._context.layers[-1]
        else:
            return None
    
    def __enter__(self):
        if SequentialModel._context is not None:
            raise RuntimeError("Nested sequential models are "
                               "not currently supported")

        SequentialModel._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert SequentialModel._context is not None
        SequentialModel._context = None