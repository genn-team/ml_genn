import numpy as np

from ml_genn import InputLayer, Layer, SequentialModel
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, IntegrateFireInput
from ml_genn.connectivity import Dense

# Create sequential model
model = SequentialModel()
with model:
    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784)
    Layer(Dense(weight=np.load("weights_0_1.npy")), 
          IntegrateFire(v_thresh=5.0))
    Layer(Dense(weight=np.load("weights_1_2.npy")), 
          IntegrateFire(v_thresh=5.0, output="spike_count"))

compiler = InferenceCompiler(dt=1.0)
compiled_model = compiler.compile(model, "simple_mnist")

# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

with compiled_model:
    # Loop through testing images
    num_correct = 0
    for i in range(testing_images.shape[0]):
        compiled_model.custom_update("Reset")
        compiled_model.set_input({input: testing_images[i] * 0.01})
        
        for t in range(100):
            compiled_model.step_time()
        
        output = compiled_model.get_output(model.layers[-1])
        print(output)
        if np.argmax(output) == testing_labels[i]:
            num_correct += 1
    print(f"{num_correct}/{testing_images.shape[0]}")
