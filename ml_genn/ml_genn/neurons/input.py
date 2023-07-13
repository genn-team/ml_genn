import numpy as np

from abc import ABC

from abc import abstractmethod

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE

class Input:
    @abstractmethod
    def set_input(self, compiled_net, pop, input):
        pass


class InputBase(Input):
    def __init__(self, var_name="Input", egp_name=None,
                 input_timesteps=1, input_dt=1,
                 **kwargs):
        super(InputBase, self).__init__(**kwargs)

        # If no EGP name is given, give error if no
        if input_timesteps > 1 and egp_name is None:
            raise RuntimeError("Neuron model does not "
                               "support time-varying input")

        self.var_name = var_name
        self.egp_name = egp_name
        self.input_timesteps = input_timesteps
        self.input_dt = input_dt

    def add_input_logic(self, neuron_model, batch_size: int, shape):
        # Replace input references in sim code with local variable reference
        neuron_model.replace_input("input")

        # If input isn't time-varying
        if self.input_timesteps == 1:
            # Add read-only input variable
            neuron_model.add_var(self.var_name, "scalar", 0.0,
                                 VarAccess_READ_ONLY_DUPLICATE)

            # Prepend sim code with code to initialize
            # local variable to input + synaptic input
            neuron_model.prepend_sim_code(
                f"const scalar input = $({self.var_name}) + $(Isyn);")
        else:
            # If batch size is 1
            flat_shape = np.prod(shape)
            if batch_size == 1:
                # Add EGP
                neuron_model.add_egp(self.egp_name, "scalar*",
                                     np.empty(self.input_timesteps,) + shape)

                # Prepend sim code with code to initialize
                # local variable to correct EGP entry + synaptic input
                neuron_model.prepend_sim_code(
                    f"""
                    const int timestep = (int)($(t) / DT);
                    const scalar input = $({self.egp_name})[($(t) * {flat_shape}) + $(id)] + $(Isyn);")
                    """)
            else:
                # Add EGP
                neuron_model.add_egp(
                    self.egp_name, "scalar*",
                    np.empty(self.input_timesteps, batch_size) + shape)

                # Prepend sim code with code to initialize
                # local variable to correct EGP entry + synaptic input
                neuron_model.prepend_sim_code(
                    f"""
                    const int timestep = (int)($(t) / DT);
                    const scalar input = $({self.egp_name})[($(t) * {flat_shape}) + $(id)] + $(Isyn);")
                    """)


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
