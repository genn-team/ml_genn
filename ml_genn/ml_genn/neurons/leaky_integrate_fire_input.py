from .input import InputBase
from .leaky_integrate_fire import LeakyIntegrateFire
from ..utils.value import InitValue


class LeakyIntegrateFireInput(LeakyIntegrateFire, InputBase):
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, tau_mem: InitValue = 20.0,
                 tau_refrac: InitValue = None, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, scale_i: bool = False,
                 input_frames=1, input_frame_time=1):
        super(LeakyIntegrateFireInput, self).__init__(
            v_thresh=v_thresh, v_reset=v_reset, v=v, tau_mem=tau_mem, 
            tau_refrac=tau_refrac, relative_reset=relative_reset, 
            integrate_during_refrac=integrate_during_refrac, scale_i=scale_i,
            egp_name="Input", input_frames=input_frames,
            input_frame_time=input_frame_time)


    def get_model(self, population, dt, batch_size):
        # Get standard integrate-and-fire model
        nm = super(LeakyIntegrateFireInput, self).get_model(population, dt,
                                                            batch_size)

        # Add input logic and replace isyn with input
        return self.create_input_model(nm, batch_size, population.shape,
                                       replace_isyn=True)
