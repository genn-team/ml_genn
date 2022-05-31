import numpy as np

from .metric import Metric


class SparseCategoricalAccuracy(Metric):
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        if y_true.shape[0] != y_pred.shape[0]:
            raise RuntimeError(f"Prediction shape:{y_pred.shape} does "
                               f"not match label shape:{y_true.shape}")

        # Add number of one-hot predictions which match labels to correct
        self.correct += np.sum((np.argmax(y_pred, axis=1) == y_true))

        # Add shape of true
        self.total += y_true.shape[0]

    def reset(self):
        self.total = 0
        self.correct = 0

    @property
    def result(self):
        if self.total == 0:
            return None
        else:
            return self.correct / self.total
