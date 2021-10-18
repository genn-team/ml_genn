from ml_genn.layers import InputType
from ml_genn.layers import IFNeurons
from ml_genn.layers import SpikeInputNeurons
from ml_genn.layers import PoissonInputNeurons
from ml_genn.layers import IFInputNeurons

class Converter(object):
    def __init__(self, input_type=InputType.SPIKE):
        self.input_type = InputType(input_type)

    def validate_tf_layer(self, tf_layer, config):
        # allow everything in test converter
        return

    def create_input_neurons(self, pre_compile_output):
        if self.input_type == InputType.SPIKE:
            return SpikeInputNeurons()
        elif self.input_type == InputType.SPIKE_SIGNED:
            return SpikeInputNeurons(signed_spikes=True)
        elif self.input_type == InputType.POISSON:
            return PoissonInputNeurons()
        elif self.input_type == InputType.POISSON_SIGNED:
            return PoissonInputNeurons(signed_spikes=True)
        elif self.input_type == InputType.IF:
            return IFInputNeurons()

    def create_neurons(self, tf_layer, pre_compile_output):
        return IFNeurons(threshold=1.0)

    def pre_convert(self, tf_model):
        pass
    
    def pre_compile(self, mlg_model):
        pass

    def post_compile(self, mlg_model):
        pass
