from enum import Enum

class InputType(Enum):
    SPIKE = 'spike'
    POISSON = 'poisson'
    IF = 'if'

class ConnectivityType(Enum):
    PROCEDURAL = 'procedural'
    SPARSE = 'sparse'
    TOEPLITZ = 'toeplitz'

class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'
