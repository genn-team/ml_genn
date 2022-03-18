class Network:
    _context = None

    def __init__(self):
        self.populations = []
        self.connections = []

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
