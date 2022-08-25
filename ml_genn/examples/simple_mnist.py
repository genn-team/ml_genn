import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, IntegrateFireInput
from ml_genn.connectivity import Dense
from ml_genn.callbacks import VarRecorder
from time import perf_counter

BATCH_SIZE = 128

# Create sequential model
network = SequentialNetwork()
with network:
    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784,
                       record_spikes=True)
    Layer(Dense(weight=np.load("weights_0_1.npy")),
          IntegrateFire(v_thresh=5.0))
    output = Layer(Dense(weight=np.load("weights_1_2.npy")),
                   IntegrateFire(v_thresh=5.0, output="spike_count"))

compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE,
                             evaluate_timesteps=100)
compiled_net = compiler.compile(network)

# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    callbacks = ["batch_progress_bar", VarRecorder(input, "V", key="in_v")]
    metrics, cb_data = compiled_net.evaluate_numpy(
        {input: testing_images * 0.01}, {output: testing_labels},
        callbacks=callbacks)
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")
    
    import matplotlib.pyplot as plt
    plt.plot(cb_data["in_v"][0])
    plt.show()
