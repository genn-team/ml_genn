from typing import Optional
from weakref import ref
from .connection import (Connection, ConnectivityInitializer,
                         SynapseInitializer)
from .population import Population, Shape, NeuronInitializer
from .sequential_network import SequentialNetwork


class InputLayer:
    """An input layer for a :class:`.SequentialNetwork`
    
    Attributes:
        population:             weak reference to underlying
                                :class:`.Population` object.

    Args:
        neuron:                 Neuron model to use (typically derived from
                                :class:`.neuron_models.Input`)
        shape:                  Shape of layer
        record_spikes:          Should spikes from this layer be recorded? 
                                This is required to use 
                                :class:`.callbacks.SpikeRecorder` 
                                with this layer.
        record_spike_events:    Should spike-like events from this layer be 
                                recorded? This is required to use 
                                :class:`.callbacks.SpikeRecorder` 
                                with this layer.
        name:                   Name of layer (only really used 
                                for debugging purposes)
    """
    def __init__(self, neuron: NeuronInitializer, shape: Shape = None,
                 record_spikes: bool = False,
                 record_spike_events: bool = False,
                 name: Optional[str] = None):
        # Create population and store weak reference in class
        population = Population(
            neuron, shape=shape, record_spikes=record_spikes,
            record_spike_events=record_spike_events,
            name=name, add_to_model=False)
        self.population = ref(population)

        SequentialNetwork._add_input_layer(self, population)


class Layer:
    """An layer for a :class:`.SequentialNetwork`
    
    Attributes:
        population:             weak reference to underlying
                                :class:`.Population` object.

    Args:
        connectivity:           Connectivity to connect layer to previous
        neuron:                 Neuron model to use
        synapse:                What type of synapse dynamics to 
                                apply to input delivered to this layers neurons
        shape:                  Shape of layer
        record_spikes:          Should spikes from this layer be recorded? 
                                This is required to use 
                                :class:`.callbacks.SpikeRecorder` 
                                with this layer.
        record_spike_events:    Should spike-like events from this layer be 
                                recorded? This is required to use 
                                :class:`.callbacks.SpikeRecorder` 
                                with this layer.
        name:                   Name of layer (only really used 
                                for debugging purposes)
        max_delay_steps:        Maximum number of delay steps this connection
                                supports. Only required when learning delays
                                or using heterogeneous delay initialiser from
                                which maximum delay cannot be inferred
    """
    def __init__(self, connectivity: ConnectivityInitializer,
                 neuron: NeuronInitializer, shape: Shape = None,
                 synapse: SynapseInitializer = "delta",
                 record_spikes: bool = False,
                 record_spike_events: bool = False,
                 max_delay_steps: Optional[int] = None,
                 name: Optional[str] = None):
        # Create population and store weak reference in class
        population = Population(
            neuron, shape=shape, record_spikes=record_spikes,
            record_spike_events=record_spike_events,
            name=name, add_to_model=False)
        self.population = ref(population)

        # If there are any preceding layers, also create
        # connection and store weak reference in class
        prev_layer = SequentialNetwork.get_prev_layer()
        if prev_layer is not None:
            connection = Connection(prev_layer.population(), population,
                                    connectivity=connectivity,
                                    synapse=synapse,
                                    max_delay_steps=max_delay_steps,
                                    add_to_model=False)
            self.connection = ref(connection)
        else:
            connection = None
            self.connection = None

        SequentialNetwork._add_layer(self, population, connection)
