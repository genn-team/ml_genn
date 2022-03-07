from enum import Enum

class ConverterType(Enum):
    SIMPLE = 'simple'
    DATA_NORM = 'data-norm'
    SPIKE_NORM = 'spike-norm'
    FEW_SPIKE = 'few-spike'
