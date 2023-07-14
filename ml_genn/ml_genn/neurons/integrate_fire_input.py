from .input import InputBase
from .integrate_fire import IntegrateFire


class IntegrateFireInput(IntegrateFire, InputBase):
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, input_timesteps=1, input_step=1):
        super(IntegrateFireInput, self).__init__(
            v_thresh=v_thresh, v_reset=v_reset, v=v, egp_name="Input",
            input_timesteps=input_timesteps, input_step=input_step)


    def get_model(self, population, dt, batch_size):
        # Get standard integrate-and-fire model
        nm = super(IntegrateFireInput, self).get_model(population, dt,
                                                       batch_size)

        # Replace standard Isyn input with reference
        # to local variable, add input logic and return
        nm.replace_input("input")
        self.add_input_logic(nm, batch_size: population.shape)
        return nm
