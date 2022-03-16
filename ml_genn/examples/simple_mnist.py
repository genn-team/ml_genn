import numpy as np
from os import path

from ml_genn import InputLayer, Layer, SequentialModel
from ml_genn.compilers import Compiler
from ml_genn.neurons import IntegrateFire, IntegrateFireInput
from ml_genn.connectivity import Dense

# Load weights
weights = []
while True:
    filename = "weights_%u_%u.npy" % (len(weights), len(weights) + 1)
    if path.exists(filename):
        weights.append(np.load(filename))
    else:
        break

# Create sequential model
model = SequentialModel()
with model:
    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784)
    for w in weights:
        Layer(Dense(weight=w), IntegrateFire(v_thresh=5.0))

compiler = Compiler(dt=0.1, prefer_in_memory_connect=True)
compiled_model = compiler.compile(model, "brunel")

# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

with compiled_model:
    # Loop through testing images
    for i in range(testing_images.shape[0]):
        # **TODO** handle weak ref
        compiled_model.set_input({input.population(): testing_images[i] * 0.01})
        
        for t in range(100):
            compiled_model.step_time()