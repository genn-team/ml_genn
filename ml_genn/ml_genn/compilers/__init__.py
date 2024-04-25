"""Compilation in mlGeNN refers to the process of converting 
:class:`ml_genn.Network` and :class:`ml_genn.SequentialNetwork`
objects into a GeNN model which can be simulated. This module contains a
variety of compiler classes which create GeNN models for inference and
training as well as several compiled model classes which can be subsequently
used to interact with the GeNN models.
"""
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from .compiled_training_network import CompiledTrainingNetwork
from .eprop_compiler import EPropCompiler
from .event_prop_compiler import EventPropCompiler
from .few_spike_compiler import CompiledFewSpikeNetwork, FewSpikeCompiler
from .inference_compiler import CompiledInferenceNetwork, InferenceCompiler

__all__ = ["Compiler", "CompiledFewSpikeNetwork", "CompiledInferenceNetwork",
           "CompiledNetwork", "CompiledTrainingNetwork", "EPropCompiler", 
           "EventPropCompiler", "FewSpikeCompiler", "InferenceCompiler"]
