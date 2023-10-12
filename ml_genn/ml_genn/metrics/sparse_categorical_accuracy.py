import numpy as np

from .metric import Metric
from ..communicators import Communicator

class SparseCategoricalAccuracy(Metric):
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred, communicator: Communicator):
        y_true = np.asarray(y_true)
        if y_true.shape[0] != y_pred.shape[0]:
            raise RuntimeError(f"Prediction shape:{y_pred.shape} does "
                               f"not match label shape:{y_true.shape}")

        # Add number of one-hot predictions which match labels to correct
        batch_correct = np.sum((np.argmax(y_pred, axis=1) == y_true))

        # Add shape of true
        batch_total = y_true.shape[0]

        # If a communicator is provided, sum number correct and total across batch
        if communicator is not None:
            batch_correct = communicator.reduce_sum(batch_correct)
            batch_total = communicator.reduce_sum(batch_correct)

        # Add total size and number correct in batch to totals
        self.correct += batch_correct
        self.total += batch_total

    def reset(self):
        self.total = 0
        self.correct = 0

    @property
    def result(self):
        if self.total == 0:
            return None
        else:
            return self.correct / self.total
