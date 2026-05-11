from numbers import Number
from typing import Tuple, Union
from .regulariser import Regulariser


class SpikeCount(Regulariser):
    """Implementation of a regulariser based on neuron spike count
    Args:
        strength:   Regularisation strength, single value or tuple,
                    if tuple, (strength for undershoot of hidden
                    spike number, strength for overshoot)
        target:     Target number of spikes"""
    def __init__(self, strength: Union[float, Tuple[float, float]],
                 target: float):
        self.target = target
        
        # If strength is specified as a single 
        # float, use for both upper and lower
        if isinstance(strength, Number):
            self.strength_lower = strength
            self.strength_upper = strength
        # Otherwise, unpack
        else:
            self.strength_lower, self.strength_upper = strength