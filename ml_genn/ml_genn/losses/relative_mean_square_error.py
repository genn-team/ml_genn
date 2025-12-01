from .loss import Loss


class RelativeMeanSquareError(Loss):
    """Computes the mean squared error between prediction of incorrect 
    outputs and correct output when there are two or more label classes, 
    specified as integers."""
    def __init__(self, delta: float):
        self.delta = delta

    @property
    def ground_truth(self) -> str:
        return "example_label"
