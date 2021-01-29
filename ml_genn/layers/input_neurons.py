from ml_genn.layers.base_neurons import BaseNeurons

class InputNeurons(BaseNeurons):

    def set_input_batch(self, data_batch):
        self.nrn.vars['input'].view[:] = data_batch.flatten()
        self.nrn.push_state_to_device()
