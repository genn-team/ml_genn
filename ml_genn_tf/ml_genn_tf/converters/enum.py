from enum import Enum

class InputType(Enum):
    SPIKE = 'spike'
    POISSON = 'poisson'
    IF = 'if'

class ConverterType(Enum):
    SIMPLE = 'simple'
    DATA_NORM = 'data-norm'
    SPIKE_NORM = 'spike-norm'
    FEW_SPIKE = 'few-spike'
