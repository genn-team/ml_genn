import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel


class ModulationMeanSquareError(Loss):
    """Computes the mean squared error between target and (target0 + prediction)"""
    # irrelevant for eventprop compiler, *TODO* do something for eprop?
    def add_to_neuron(self, model: NeuronModel, shape, 
                      batch_size: int, example_timesteps: int):
        # Add extra global parameter to store Y* (target) and Y0 ("source" to be modulated) throughout example
        flat_shape = np.prod(shape[1:])
        egp_size = (example_timesteps * batch_size * flat_shape)
        model.add_egp("YTrue", "scalar*",
                      np.empty(egp_size, dtype=np.float32))
        model.add_egp("YSource", "scalar*",
                      np.empty(egp_size, dtype=np.float32))

        # Add sim-code to read out correct yTrue value *TODO* NO IDEA WHETHER THE BELOW MAKES SENSE FOR EPROP
        model.append_sim_code(
            f"""
            const unsigned int timestep = (int)round(t / dt);
            const unsigned int index = (batch * {example_stimesteps} * num_neurons)
                                       + (timestep * num_neurons) + id;
            const scalar yTrue = YTrue[index];
            """)
 
            
    def set_target(self, genn_pop, y_true, shape, batch_size: int, 
                   example_timesteps: int):
        # Check shape
        expected_shape = (batch_size, 2, example_timesteps) + shape
        y_true = np.asarray(y_true)
        if y_true.shape != expected_shape:
            raise RuntimeError(f"Shape of target data for ModulationMeanSquareError "
                               f"loss should be {expected_shape}")

        # Copy flattened y_true elements into views
        genn_pop.extra_global_params["YSource"].view[:] = y_true[:,0].flatten()
        assert(len(y_true[:,0].flatten()) == batch_size * example_timesteps * np.prod(shape))
        genn_pop.extra_global_params["YTrue"].view[:] = y_true[:,1].flatten()

        # Push YSource and YTrue to device
        genn_pop.extra_global_params["YSource"].push_to_device()
        genn_pop.extra_global_params["YTrue"].push_to_device()

