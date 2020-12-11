import numpy as np
from tensor_genn.layers.base_neurons import BaseNeurons

class InputNeurons(BaseNeurons):

    def set_input_batch(self, data_batch):
        assert(data_batch.shape[0] <= len(self.nrn))
        assert(data_batch.shape[1:] == self.shape)

        # Set inputs
        for batch_i in range(len(self.nrn)):
            if batch_i < data_batch.shape[0]:
                self.nrn[batch_i].vars['input'].view[:] = data_batch[batch_i].flatten()
            else:
                self.nrn[batch_i].vars['input'].view[:] = np.zeros(np.prod(self.shape))
            self.nrn[batch_i].push_state_to_device()
