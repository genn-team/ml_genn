import numpy as np

from abc import ABC

from abc import abstractmethod


class Input:
    @abstractmethod
    def set_input(self, compiled_net, pop, input):
        pass


class InputBase(Input):
    def __init__(self, var_name="Input", **kwargs):
        super(InputBase, self).__init__(**kwargs)

        self.var_name = var_name

    def set_input(self, genn_pop, batch_size: int, shape, input):
        # Ensure input is something convertable to a numpy array
        input = np.asarray(input)

        # If batch size is 1
        if batch_size == 1:
            # Give error if shapes don't match
            if input.shape != shape and (input.shape[0] != 1
                                         or input.shape[1:] != shape):
                raise RuntimeError(f"Input shape {input.shape} does not match "
                                   f"population shape {shape}")

            # Flatten input and copy into view
            genn_pop.vars[self.var_name].view[:] = input.flatten()
        # Otherwise
        else:
            # If non-batch dimensions of input don't match shape
            if input.shape[1:] != shape:
                raise RuntimeError(f"Input shape {input.shape[1:]} does not "
                                   f"match population shape {shape}")

            # Reshape input into batches of flattened data
            batched_input = np.reshape(input, (-1, np.prod(shape)))

            # If we have a full batch
            input_batch_size = batched_input.shape[0]
            if input_batch_size == batch_size:
                genn_pop.vars[self.var_name].view[:] = batched_input
            # Otherwise, pad up to full batch
            elif input_batch_size < batch_size:
                genn_pop.vars[self.var_name].view[:] = np.pad(
                    batched_input,
                    ((0, batch_size - input_batch_size), (0, 0)))
            else:
                raise RuntimeError(f"Input batch size {input_batch_size} does "
                                   f"not match batch size {batch_size}")

        # Push variable to device
        genn_pop.push_var_to_device(self.var_name)
