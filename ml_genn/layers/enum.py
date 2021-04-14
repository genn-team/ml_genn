from enum import Enum

class ConnectivityType(Enum):
    PROCEDURAL = 'procedural'
    SPARSE = 'sparse'

class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'