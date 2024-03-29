from .input import InputBase
from .integrate_fire import IntegrateFire
from ..utils.value import InitValue


class IntegrateFireInput(IntegrateFire, InputBase):
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, input_frames=1, input_frame_timesteps=1):
        super(IntegrateFireInput, self).__init__(
            v_thresh=v_thresh, v_reset=v_reset, v=v,
            egp_name="Input", input_frames=input_frames,
            input_frame_timesteps=input_frame_timesteps)


    def get_model(self, population, dt, batch_size):
        # Get standard integrate-and-fire model
        nm = super(IntegrateFireInput, self).get_model(population, dt,
                                                       batch_size)

        # Add input logic and replace isyn with input
        return self.create_input_model(nm, batch_size, population.shape,
                                       replace_input="$(Isyn)")
