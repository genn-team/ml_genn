from weakref import ref
from .connection import Connection
from .population import Population, Shape
from .sequential_model import SequentialModel
from .connectivity import Connectivity
from .neurons import Neuron
from .synapses import Synapse

class InputLayer:
    def __init__(self, neuron: Neuron, shape=None):
        # Create population and store weak reference in class
        population = Population(neuron, shape=shape, add_to_model=False)
        self.population = ref(population)
        
        SequentialModel.add_input_layer(self, population)

class Layer:
    def __init__(self, connectivity: Connectivity, neuron: Neuron, shape=None, synapse="delta"):
        # Create population and store weak reference in class
        population = Population(neuron, shape=shape, add_to_model=False)
        self.population = ref(population)
        
        # If there are any preceding layers, also create  
        # connection and store weak reference in class
        prev_layer = SequentialModel.get_prev_layer()
        if prev_layer is not None:
            connection = Connection(prev_layer.population(), population, 
                                    connectivity=connectivity, synapse=synapse,
                                    add_to_model=False)
            self.connection = ref(connection)
        else:
            connection = None
            self.connection = None
        
        SequentialModel.add_layer(self, population, connection)
