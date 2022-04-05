import numpy as np

class SparseCategoricalAccuracy:
    def __init__(self):
        self.reset()
    
    def __call__(self, y_true, y_pred):
        print(y_true.shape, y_pred.shape)
    
        # Add to number correct
        self.correct += np.sum((np.argmax(y_pred, axis=1) == y_true))
        
        self.total += y_true.shape[0]
    
    def reset(self):
        self.total = 0
        self.correct = 0
