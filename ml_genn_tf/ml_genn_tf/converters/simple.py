import tensorflow as tf

from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import (BinarySpikeInput, IntegrateFire,
                             IntegrateFireInput, PoissonInput)
from .converter import Converter
from .enum import InputType

class Simple(Converter):
    def __init__(self, evaluate_timesteps, signed_input: bool = False,
                 input_type: InputType = InputType.POISSON):
        self.evaluate_timesteps = evaluate_timesteps
        self.signed_input = signed_input
        self.input_type = InputType(input_type)

    def create_input_neurons(self, pre_compile_output):
        if self.input_type == InputType.SPIKE:
            return BinarySpikeInput(signed_spikes=self.signed_input)
        elif self.input_type == InputType.POISSON:
            return PoissonInput(signed_spikes=self.signed_input)
        elif self.input_type == InputType.IF:
            return IntegrateFireInput()

    def create_neurons(self, tf_layer, pre_compile_output, is_output):
        return IntegrateFire(v_thresh=1.0, 
                             readout="spike_count" if is_output else None)

    def pre_convert(self, tf_model):
        pass

    def pre_compile(self, mlg_network):
        pass

    def create_compiler(self, **kwargs):
        return InferenceCompiler(evaluate_timesteps=self.evaluate_timesteps,
                                 **kwargs)
