from enum import Enum

class InputType(Enum):
    SPIKE = 'spike'
    POISSON = 'poisson'
    POISSON_SIGNED = 'poisson_signed'
    IF = 'if'

class ConnectionType(Enum):
    PROCEDURAL = 'procedural'
    SPARSE = 'sparse'

class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'
