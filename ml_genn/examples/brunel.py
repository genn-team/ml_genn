from ml_genn import Connection, Network, Population
from ml_genn.compilers import Compiler
from ml_genn.neurons import LeakyIntegrateFire
from ml_genn.connectivity import FixedProbability

neuron = LeakyIntegrateFire(v_thresh=20.0, v_reset=10.0, tau_mem=20.0, tau_refrac=2.0, relative_reset=False)
e_conn = FixedProbability(p=0.1, weight=0.1)
i_conn = FixedProbability(p=0.1, weight=-0.5)

network = Network()
with network:
    e = Population(neuron, 7500)
    i = Population(neuron, 2500)
    
    ee = Connection(e, e, e_conn)
    ei = Connection(e, i, e_conn)
    ii = Connection(i, i, i_conn)
    ie = Connection(i, e, i_conn)

compiler = Compiler(dt=0.1, prefer_in_memory_connect=True)
compiled_network = compiler.compile(network, "brunel")