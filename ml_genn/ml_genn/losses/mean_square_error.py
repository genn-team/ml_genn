from .loss import Loss

class MeanSquareError(Loss):
    """Computes the mean squared error between labels and prediction"""
    @property
    def prediction(self) -> str:
        return "timestep_value"
