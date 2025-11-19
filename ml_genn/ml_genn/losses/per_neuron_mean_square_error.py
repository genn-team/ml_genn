from .loss import Loss


class PerNeuronMeanSquareError(Loss):
    """Computes the mean squared error between labels and prediction"""
    @property
    def ground_truth(self) -> str:
        return "example_value"

