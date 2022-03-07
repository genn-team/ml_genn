#from . import Population

class Model:
    _context = None

    def __init__(self):
        self.populations = []
        self.connections = []

    @staticmethod
    def add_population(pop):
        if Model._context is None:
            raise RuntimeError("Population must be created "
                               "inside a ``with model:`` block")
        Model._context.populations.append(pop)

    @staticmethod
    def add_connection(conn):
        if Model._context is None:
            raise RuntimeError("Connection must be created "
                               "inside a ``with model:`` block")
        Model._context.connections.append(conn)

    def __enter__(self):
        if Model._context is not None:
            raise RuntimeError("Nested models are not currently supported")

        Model._context = self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert Model._context is not None
        Model._context = None