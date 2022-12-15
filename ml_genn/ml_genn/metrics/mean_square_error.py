import numpy as np

from .metric import Metric


class MeanSquareError(Metric):
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        if y_true.shape != y_pred.shape:
            raise RuntimeError(f"Prediction shape:{y_pred.shape} does "
                               f"not match label shape:{y_true.shape}")

        # Add mean square error between truth and prediction
        self.sum_mse += np.mean(np.square(y_true - y_pred))

        # Add shape of true
        self.total += y_true.shape[0]

    def reset(self):
        self.sum_mse = 0.0
        self.total = 0

    @property
    def result(self):
        if self.total == 0:
            return None
        else:
            return self.sum_mse / self.total
