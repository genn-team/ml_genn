import numpy as np

from collections import namedtuple
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from typing import Sequence, Union
from .input_base import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Population

genn_model = {
    "extra_global_params": [("Input", "uint32_t*")],
 
    "threshold_condition_code": """
    $(Input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    "is_auto_refractory_required": False}


class AEStreamInput(Neuron, InputBase):
    def __init__(self):
        super(SpikeInput, self).__init__()

    def set_input(self, genn_pop, batch_size: int, shape, input):
        input.read_genn(genn_pop.extra_global_params["Input"].view)
        genn_pop.push_extra_global_param_to_device("Input")
    
    def get_model(self, population: "Population", dt: float):
        num_neurons = np.prod(population.shape)
        num_words = (num_neurons + 31) // 32
        return NeuronModel(genn_model, None, {}, {},
                           {"Input": np.zeros(num_words, dtype=np.uint32)})
