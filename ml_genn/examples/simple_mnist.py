import numpy as np
import matplotlib.pyplot as plt

from ml_genn import InputLayer, Layer, SequentialModel
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, IntegrateFireInput
from ml_genn.connectivity import Dense

from time import perf_counter

BATCH_SIZE = 128

# Create sequential model
model = SequentialModel()
with model:
    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784)
    Layer(Dense(weight=np.load("weights_0_1.npy")), 
          IntegrateFire(v_thresh=5.0))
    output = Layer(Dense(weight=np.load("weights_1_2.npy")), 
                   IntegrateFire(v_thresh=5.0, output="spike_count"))

compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE, 
                             evaluate_timesteps=100)
compiled_model = compiler.compile(model, "simple_mnist")

# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

with compiled_model:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    accuracy = compiled_model.evaluate_numpy({input: testing_images * 0.01},
                                             {output: testing_labels})
    end_time = perf_counter()
    print(f"Accuracy = {100 * accuracy[model.layers[-1]]}%")
    print(f"Time = {end_time - start_time}s")
    
