from ..utils import InitValue, Value

class Connectivity:
    def __init__(self, weight: [InitValue], delay: [InitValue]):
        self.weight = Value(weight)
        self.delay = Value(delay)
