from enum import Enum

class InputType(Enum):
    SPIKE = 'spike'
    SPIKE_SIGNED = 'spike_signed'
    POISSON = 'poisson'
    POISSON_SIGNED = 'poisson_signed'
    IF = 'if'

class ConnectivityType(Enum):
    PROCEDURAL = 'procedural'
    SPARSE = 'sparse'

class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'
