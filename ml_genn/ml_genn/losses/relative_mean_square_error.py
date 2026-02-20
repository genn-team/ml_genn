from typing import Any, Optional
from .loss import Loss


class RelativeMeanSquareError(Loss):
    """Computes the mean squared error between prediction of incorrect 
    outputs and correct output when there are two or more label classes, 
    specified as integers."""
    def __init__(self, delta: float, record_key: Optional[Any] = None):
        super().__init__(record_key)

        self.delta = delta

    @property
    def ground_truth(self) -> str:
        return "example_label"
