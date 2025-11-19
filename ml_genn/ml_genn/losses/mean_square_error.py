from .loss import Loss

class MeanSquareError(Loss):
    """Computes the mean squared error between labels and prediction"""
    @property
    def ground_truth(self) -> str:
        return "timestep_value"
