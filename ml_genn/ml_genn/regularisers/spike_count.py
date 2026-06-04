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
        self.target = float(target)
        
        # Try and unpack tuple of strengths
        try:
            self.strength_lower, self.strength_upper = strength
        # If it's not unpackable, use strength for lower and upper
        # **NOTE** ValueErrors relating to wrong number of values still propagate
        except TypeError:
            self.strength_lower = strength
            self.strength_upper = strength
        
        # Ensure strengths are convertable to float
        self.strength_lower = float(self.strength_lower)
        self.strength_upper = float(self.strength_upper)