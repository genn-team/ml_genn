from __future__ import annotations

import numpy as np

from abc import ABC

from abc import abstractmethod
from copy import deepcopy

from pygenn import VarAccess


def _replace_neuron_code(nm, source, target):
    nm.replace_sim_code(source, target)
    nm.replace_threshold_condition_code(source, target)
    nm.replace_reset_code(source, target)


class Input:
    """Base class for all types of input neuron"""
    @abstractmethod
    def set_input(self, genn_pop, batch_size: int, shape, input):
        """
        Copy provided data to GPU.
        
        Args:
            genn_pop:   GeNN ``NeuronGroup`` object population has been
                        compiled into
            batch_size: Batch size of compiled network
            shape:      Shape of input population
            input:      Input data
        """
        pass


class InputBase(Input):
    """Base class for all types of input neuron for non-spiking datasets.
    
    Args:
        var_name:               Name of state variable to add to 
                                model to hold current static input
        egp_name:               Name of Extra Global Parameter to add to 
                                model to hold current time-varying input
        input_frames:           How many frames does each input have?
        input_frame_timesteps:  How many timesteps should each frame of 
                                input be presented for?
     """
    def __init__(self, var_name="Input", egp_name=None,
                 input_frames=1, input_frame_timesteps=1,
                 **kwargs):
        super(InputBase, self).__init__(**kwargs)

        # If input has multiple frames but no EGP name is given, give error
        if input_frames > 1 and egp_name is None:
            raise RuntimeError("Neuron model does not "
                               "support time-varying input")

        self.var_name = var_name
        self.egp_name = egp_name
        self.input_frames = input_frames
        self.input_frame_timesteps = input_frame_timesteps

    def create_input_model(self, base_model, batch_size: int, shape,
                           replace_input: str = None):
        """Convert standard neuron model into input neuron model.
        
        Args:
            base_model:     Standard neuron model
            batch_size:     Batch size of compiled network
            shape:          Shape of input population
            replace_input:  Name of variable in neuron model code 
                            to replace with input (typically Isyn)
        """
        # Make copy of model
        nm_copy = deepcopy(base_model)

        # If input isn't time-varying
        if self.input_frames == 1:
            # Add read-only input variable
            nm_copy.add_var(self.var_name, "scalar", 0.0,
                            VarAccess.READ_ONLY_DUPLICATE)

            # Replace input with reference to new variable
            nm_copy.replace_sim_code(replace_input, self.var_name)
            nm_copy.replace_threshold_condition_code(replace_input, self.var_name)
            nm_copy.replace_reset_code(replace_input, self.var_name)
        else:
            # Check there isn't a variable which conflicts with EGP
            assert not nm_copy.has_var(self.egp_name)

            # Replace input with references to local variable
            # **NOTE** do this first so it doesn't modify the new code we add!
            _replace_neuron_code(nm_copy, replace_input, "input")

            # If batch size is 1
            flat_shape = np.prod(shape)
            if batch_size == 1:
                # Add EGP
                nm_copy.add_egp(self.egp_name, "scalar*",
                                np.empty((self.input_frames,) + shape,
                                         dtype=np.float32))

                # Prepend sim code with code to initialize
                # local variable to correct EGP entry + synaptic input
                nm_copy.prepend_sim_code(
                    f"""
                    const int timestep = min((int)(t / ({self.input_frame_timesteps} * dt)), {self.input_frames - 1});
                    const scalar input = {self.egp_name}[(timestep * {flat_shape}) + id];
                    """)
            else:
                # Add EGP
                nm_copy.add_egp(
                    self.egp_name, "scalar*",
                    np.empty((self.input_frames, batch_size) + shape,
                             dtype=np.float32))

                # Prepend sim code with code to initialize
                # local variable to correct EGP entry + synaptic input
                nm_copy.prepend_sim_code(
                    f"""
                    const int timestep = min((int)(t / ({self.input_frame_timesteps} * dt)), {self.input_frames - 1});
                    const scalar input = {self.egp_name}[(batch * {flat_shape * self.input_frames}) + (timestep * {flat_shape}) + id];
                    """)
        return nm_copy

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

        # If input isn't time-varying
        if self.input_frames == 1:
            # If batch size is 1
            if batch_size == 1:
                # Check input shape either has no batch
                # dimension or it has a length of 1
                if (len(input_batch_time_dims) != 0
                    and (len(input_batch_time_dims) != 1
                         or input_batch_time_dims[0] != 1)):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Flatten input and copy into view
                genn_pop.vars[self.var_name].view[:] = input.flatten()
            # Otherwise
            else:
                # Check input shape has batch dimension
                # and this is less than or equal to batch size
                if (len(input_batch_time_dims) != 1
                    or input_batch_time_dims[0] > batch_size):
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
            genn_pop.vars[self.var_name].push_to_device()
        # Otherwise
        else:
            # Check time dimension matches
            if (len(input_batch_time_dims) == 0
                or input_batch_time_dims[-1] != self.input_frames):
                raise RuntimeError(f"Input shape {input.shape} does not "
                                   f"match timesteps {self.input_frames}")

            # If batch size is 1
            egp_view = genn_pop.extra_global_params[self.egp_name].view
            if batch_size == 1:
                # Check input shape either has no batch
                # dimension or it has a length of 1
                if (len(input_batch_time_dims) != 1
                    and (len(input_batch_time_dims) != 2
                         or input_batch_time_dims[0] != 1)):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Flatten input and copy into view
                egp_view[:] = input.flatten()
            # Otherwise
            else:
                # Check input shape has batch dimension
                # and this is less than or equal to batch size
                if (len(input_batch_time_dims) != 2
                    or input_batch_time_dims[0] > batch_size):
                    raise RuntimeError(f"Input shape {input.shape} does "
                                       f"not match batch size {batch_size}")

                # Reshape input into batches of flattened data
                input_size = self.input_frames * np.prod(shape)
                batched_input = np.reshape(input, (-1, input_size))

                # If we have a full batch
                input_batch_size = batched_input.shape[0]
                if input_batch_size == batch_size:
                    egp_view[:] = batched_input.flatten()

                # Otherwise, pad up to full batch
                else:
                    egp_view[:] = np.pad(batched_input,
                                         ((0, batch_size - input_batch_size),
                                          (0, 0))).flatten()

            # Push variable to device
            genn_pop.extra_global_params[self.egp_name].push_to_device()

