from enum import Enum

class InputType(Enum):
    SPIKE = 'spike'
    POISSON = 'poisson'
    IF = 'if'