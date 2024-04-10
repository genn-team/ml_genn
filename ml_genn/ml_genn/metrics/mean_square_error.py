import numpy as np

from typing import Optional
from .metric import Metric
from ..communicators import Communicator

class MeanSquareError(Metric):
    """Computes the mean squared error between labels and prediction"""
    def __init__(self):
        self.reset()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray,
               communicator: Optional[Communicator]):
        y_true = np.asarray(y_true)
        if y_true.shape != y_pred.shape:
            raise RuntimeError(f"Prediction shape:{y_pred.shape} does "
                               f"not match label shape:{y_true.shape}")

        # Add mean square error between truth and prediction
        batch_sum_mse = np.mean(np.square(y_true - y_pred))

        # Add shape of true
        batch_total = y_true.shape[0]

        # If a communicator is provided, sum number correct and total across batch
        if communicator is not None:
            batch_sum_mse = communicator.reduce_sum(batch_sum_mse)
            batch_total = communicator.reduce_sum(batch_total)

        # Add total size and MSE sum across batch to totals
        self.sum_mse += batch_sum_mse
        self.total += batch_total

    def reset(self):
        self.sum_mse = 0.0
        self.total = 0

    @property
    def result(self):
        if self.total == 0:
            return None
        else:
            return self.sum_mse / self.total
