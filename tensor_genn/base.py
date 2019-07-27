import numpy as np

import tensorflow as tf
from pygenn import genn_model, genn_wrapper

def convert_model(tf_model, algorithm, X=None, y=None, weight_normalizer=None):
    scaled_tf_weights = None
    if weight_normalizer is not None:
        scaled_tf_weights = weight_normalizer.normalize(tf_model)
        print("Weights normalized")
    g_model = algorithm.convert(tf_model, scaled_tf_weights)
    print("Model converted")
    if X is not None and y is not None:
        print("Evaluating GeNN model")
        accuracy = algorithm.evaluate(X,y)
        print("Accuracy achieved by GeNN model: {}%".format(accuracy))

    return g_model