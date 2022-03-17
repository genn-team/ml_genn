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
    Layer(Dense(weight=np.load("weights_1_2.npy")), 
          IntegrateFire(v_thresh=5.0, output="spike_count"))

compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE)
compiled_model = compiler.compile(model, "simple_mnist")

# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

# Split into batches
num_images = testing_images.shape[0]
testing_images =  np.split(testing_images, range(BATCH_SIZE, num_images, BATCH_SIZE), axis=0)
testing_labels =  np.split(testing_labels, range(BATCH_SIZE, num_images, BATCH_SIZE), axis=0)

with compiled_model:
    # Loop through testing images
    num_correct = 0
    start_time = perf_counter()
    for image_batch, label_batch in zip(testing_images, testing_labels):
        batch_count = len(label_batch)
        compiled_model.custom_update("Reset")
        compiled_model.set_input({input: image_batch * 0.01})
        
        for t in range(100):
            compiled_model.step_time()

        output = compiled_model.get_output(model.layers[-1])
        num_correct += np.sum(np.argmax(output[:batch_count], axis=1) == label_batch)
            
    end_time = perf_counter()
    print(f"Accuracy = {(100.0 * num_correct) / num_images}%")
    print(f"Time = {end_time - start_time}s")
