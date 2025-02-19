"""Neuron models describe the dynamics and spiking behaviour of a :class:`..Population`.
"""

from .adaptive_leaky_integrate_fire import AdaptiveLeakyIntegrateFire
from .binary_spike_input import BinarySpikeInput
from .few_spike_relu import FewSpikeRelu
from .few_spike_relu_input import FewSpikeReluInput
from .input import Input
from .integrate_fire import IntegrateFire
from .integrate_fire_input import IntegrateFireInput
from .leaky_integrate import LeakyIntegrate
from .leaky_integrate_fire import LeakyIntegrateFire
from .leaky_integrate_fire_input import LeakyIntegrateFireInput
from .neuron import Neuron
from .poisson_input import PoissonInput
from .spike_input import SpikeInput

from ..utils.module import get_module_classes

default_neurons = get_module_classes(globals(), Neuron)

__all__ = ["AdaptiveLeakyIntegrateFire", "BinarySpikeInput", "FewSpikeRelu",
           "FewSpikeReluInput", "Input", "IntegrateFire", "IntegrateFireInput",
           "LeakyIntegrate", "LeakyIntegrateFire", "LeakyIntegrateFireInput", 
           "Neuron", "PoissonInput", "SpikeInput", "default_neurons"]
