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
                 input_timesteps=1, input_step=1,
                 **kwargs):
        super(InputBase, self).__init__(**kwargs)

        # If no EGP name is given, give error if no
        if input_timesteps > 1 and egp_name is None:
            raise RuntimeError("Neuron model does not "
                               "support time-varying input")

        self.var_name = var_name
        self.egp_name = egp_name
        self.input_timesteps = input_timesteps
        self.input_step = input_step

    def add_input_logic(self, neuron_model, batch_size: int, shape):
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
                # local variable tosigned_spikes correct EGP entry + synaptic input
                neuron_model.prepend_sim_code(
                    f"""
                    const int timestep = min((int)round($(t) / ({self.input_step} * DT)), {self.input_timesteps - 1});
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
                    const int timestep = min((int)round($(t) / ({self.input_step} * DT)), {self.input_timesteps - 1});
                    const scalar input = $({self.egp_name})[($(batch) * {flat_shape * self.input_timesteps}) + (timestep * {flat_shape}) + $(id)] + $(Isyn);")
                    """)


    def set_input(self, genn_pop, batch_size: int, shape, input):
        # Ensure input is something convertable to a numpy array
        input = np.asarray(input)

        # Split input shape into the bit that should match population
        # shape and the bit that should match time and batch
        input_shape_dims = input.shape[-len(shape):]
        input_batch_time_dims = input.shape[:-len(shape)]

        # Check last dimensions of input shape match population shape
        if input_shape_dims != shape:
            raise RuntimeError(f"Input shape {input.shape} does not match "
                               f"population shape {shape}")

        # If input ivar_namesn't time-varying
        if self.input_timesteps == 1:
            # If batch size is 1
            if batch_size == 1:
                # Check input shape either has no batch
                # dimension or it has a length of 1
                if (len(input_time_batch_dims) != 0
                    and (len(input_time_batch_dims) != 1
                         or input_time_batch_dims[0] != 1):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Flatten input and copy into view
                genn_pop.vars[self.var_name].view[:] = input.flatten()
            # Otherwise
            else:
                # Check input shape has batch dimension
                # and this is less than or equal to batch size
                if (len(input_time_batch_dims) != 1
                    or input_time_batch_dims[0] > batch_size):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Reshape input into batches of flattened data
                batched_input = np.reshape(input, (-1, np.prod(shape)))

                # If we have a full batch
                input_batch_size = batched_input.shape[0]
                if input_batch_size == batch_size:
                    genn_pop.vars[self.var_name].view[:] = batched_input

                # Otherwise, pad up to full batch
                else:
                    genn_pop.vars[self.var_name].view[:] = np.pad(
                        batched_input,
                        ((0, batch_size - input_batch_size), (0, 0)))

            # Push variable to device
            genn_pop.push_var_to_device(self.var_name)
        # Otherwise
        else:
            # Check time dimension matches
            if (len(input_batch_time_dims) == 0
                or input_batch_time_dims[-1] != self.input_timesteps):
                raise RuntimeError(f"Input shape {input.shape} does not "
                                   f"match timesteps {self.input_timesteps}")

            # If batch size is 1
            egp_view = genn_pop.extra_global_params[self.egp_name].view
            if batch_size == 1:
                # Check input shape either has no batch
                # dimension or it has a length of 1
                if (len(input_time_batch_dims) != 1
                    and (len(input_time_batch_dims) != 2
                         or input_time_batch_dims[0] != 1):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Flatten input and copy into view
                egp_view[:] = input.flatten()
            # Otherwise
            else:
                # Check input shape has batch dimension
                # and this is less than or equal to batch size
                if (len(input_time_batch_dims) != 2
                    or input_time_batch_dims[0] > batch_size):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Reshape input into batches of flattened data
                input_size = self.input_timesteps * np.prod(shape)
                batched_input = np.reshape(input, (-1, input_size))

                # If we have a full batch
                input_batch_size = batched_input.shape[0]
                if input_batch_size == batch_size:
                    egp_view[:] = batched_input

                # Otherwise, pad up to full batch
                else:
                    egp_view[:] = np.pad(batched_input,
                                         ((0, batch_size - input_batch_size),
                                          (0, 0)))

            # Push variable to device
            genn_pop.push_extra_global_param_to_device(self.egp_name)
