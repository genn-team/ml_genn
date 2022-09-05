import numpy as np

from ml_genn import Connection, Population, Network
from ml_genn.callbacks import VarRecorder
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam

from ml_genn_eprop import EPropCompiler

NUM_INPUT = 20
NUM_HIDDEN = 256
NUM_OUTPUT = 3

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=100), NUM_INPUT)
    hidden = Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                           tau_refrac=5.0, 
                                           relative_reset=True,
                                           integrate_during_refrac=True),
                        NUM_HIDDEN)
    output = Population(LeakyIntegrate(tau_mem=20.0, output="var"),
                        NUM_OUTPUT)
    
    # Connections
    Connection(input, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))))
    Connection(hidden, hidden, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))))
    Connection(hidden, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))))

compiler = EPropCompiler(example_timesteps=1000, losses="mean_square_error",
                         optimiser=Adam(), c_reg=3.0)
compiled_net = compiler.compile(network)