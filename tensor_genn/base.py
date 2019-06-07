import numpy as np

import tensorflow as tf
from pygenn import genn_model, genn_wrapper

def convert_model(tf_model, algorithm, X=None, y=None):
    g_model = algorithm.convert(tf_model)
    print("Model converted")
    if X is not None and y is not None:
        print("Evaluating GeNN model")
        accuracy = algorithm.evaluate(X,y)
        print("Accuracy achieved by GeNN model: {}%".format(accuracy))

    return g_model