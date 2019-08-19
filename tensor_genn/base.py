import numpy as np

import tensorflow as tf
from pygenn import genn_model, genn_wrapper

def convert_model(tf_model, algorithm, X=None, y=None, weight_normalizer=None, save_example_spikes=[]):
    # Normalize weights if necessary
    scaled_tf_weights = None
    if weight_normalizer is not None:
        scaled_tf_weights = weight_normalizer.normalize(tf_model)
        print("Weights normalized")

    # Convert model according to algorithm
    g_model = algorithm.convert(tf_model, scaled_tf_weights)
    print("Model converted")

    # Evaluate GeNN model on test set if data has been provided
    if X is not None and y is not None:
        print("Evaluating GeNN model")
        accuracy, spike_ids, spike_times, neuron_pops, syn_pops = algorithm.evaluate(X,y,save_example_spikes)
        print("Accuracy achieved by GeNN model: {}%".format(accuracy))

    return g_model, spike_ids, spike_times, neuron_pops, syn_pops