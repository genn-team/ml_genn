from .loss import Loss

class SparseCategoricalCrossentropy(Loss):
    """Computes the crossentropy between labels and prediction 
    when there are two or more label classes, specified as integers."""
    @property
    def ground_truth(self) -> str:
        return "example_label"
