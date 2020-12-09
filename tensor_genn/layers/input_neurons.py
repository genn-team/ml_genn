import numpy as np
from tensor_genn.layers.base_neurons import BaseNeurons

class InputNeurons(BaseNeurons):

    def set_input_batch(self, data_batch):
        n_samples = data_batch.shape[0]
        if n_samples > len(self.nrn):
            raise ValueError('sample count {} > batch size {}'.format(n_samples, len(self.nrn)))
        sample_size = np.prod(data_batch.shape[1:])
        input_size = np.prod(self.shape)
        if sample_size != input_size:
            raise ValueError('sample size {} != input size {}'.format(sample_size, input_size))

        # Set inputs
        for batch_i in range(len(self.nrn)):
            if batch_i < n_samples:
                self.nrn[batch_i].vars['input'].view[:] = data_batch[batch_i].flatten()
            else:
                self.nrn[batch_i].vars['input'].view[:] = np.zeros(input_size)
            self.nrn[batch_i].push_state_to_device()
