from typing import Optional
from weakref import ref
from .connection import (Connection, ConnectivityInitializer,
                         SynapseInitializer)
from .population import Population, Shape, NeuronInitializer
from .sequential_network import SequentialNetwork


class InputLayer:
    def __init__(self, neuron: NeuronInitializer, shape: Shape = None,
                 record_spikes: bool = False, name: Optional[str] = None):
        # Create population and store weak reference in class
        population = Population(neuron, shape=shape,
                                record_spikes=record_spikes,
                                name=name, add_to_model=False)
        self.population = ref(population)

        SequentialNetwork._add_input_layer(self, population)


class Layer:
    def __init__(self, connectivity: ConnectivityInitializer,
                 neuron: NeuronInitializer, shape: Shape = None,
                 synapse: SynapseInitializer = "delta",
                 record_spikes: bool = False, name: Optional[str] = None):
        # Create population and store weak reference in class
        population = Population(neuron, shape=shape,
                                record_spikes=record_spikes,
                                name=name, add_to_model=False)
        self.population = ref(population)

        # If there are any preceding layers, also create
        # connection and store weak reference in class
        prev_layer = SequentialNetwork.get_prev_layer()
        if prev_layer is not None:
            connection = Connection(prev_layer.population(), population,
                                    connectivity=connectivity,
                                    synapse=synapse, add_to_model=False)
            self.connection = ref(connection)
        else:
            connection = None
            self.connection = None

        SequentialNetwork._add_layer(self, population, connection)
