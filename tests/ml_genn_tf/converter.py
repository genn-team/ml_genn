import numpy as np

from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import BinarySpikeInput, IntegrateFire

from ml_genn_tf.converters import Converter

class Converter(Converter):
    def validate_tf_layer(self, tf_layer, config):
        # allow everything in test converter
        return

    def create_input_neurons(self, pre_compile_output):
        return BinarySpikeInput()

    def create_neurons(self, tf_layer, pre_convert_output, is_output):
        if is_output:
            v_thresh = np.float64(np.finfo(np.float32).max)
            return IntegrateFire(v_thresh=v_thresh, readout="var")
        else:
            return IntegrateFire(v_thresh=1.0)

    def create_compiler(self, **kwargs):
        return InferenceCompiler(evaluate_timesteps=2, **kwargs)