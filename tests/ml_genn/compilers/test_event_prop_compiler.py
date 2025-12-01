import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.neurons import LeakyIntegrate, SpikeInput, LeakyIntegrateFire
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
from ml_genn.callbacks import SpikeRecorder, VarRecorder

from ml_genn.utils.data import preprocess_spikes


def test_parameter_learning():
    # Build spike trains
    ind = [[0, 1],[1, 0]]
    time = [[0, 10],[0, 10]]
    spikes = [preprocess_spikes(np.asarray(t), np.asarray(i), 2)
              for t, i in zip(time, ind)]
    labels = [0, 1]
   
    w_in_hid = [[4, 0, 4, 0],
                [0, 4, 0, 4 ]]
    w_hid_out = [[4, 0],
                [4, 0],
                [0, 4],
                [0, 4]]

    network = SequentialNetwork()
    with network:
        input = InputLayer(SpikeInput(max_spikes=2), 2)
        hidden = Layer(Dense(w_in_hid),  LeakyIntegrateFire(tau_mem=10.0), 
                       4, Exponential(5))
        output = Layer(Dense(w_hid_out), 
                       LeakyIntegrate(tau_mem=10.0, readout="max_var"),
                       2, Exponential(5))

    compiler = EventPropCompiler(example_timesteps=100,
                                 losses="sparse_categorical_crossentropy",
                                 batch_size=1)
    compiled_net = compiler.compile(network, optimisers= {hidden: {"tau_mem": Adam(0.01)}})

    with compiled_net:
        metrics, _  = compiled_net.train({input: spikes}, {output: labels}, 
                                         shuffle=False, num_epochs=40)
        
        assert metrics[output].result == 1.0
