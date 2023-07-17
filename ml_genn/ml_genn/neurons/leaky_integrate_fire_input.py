from .input import InputBase
from .leaky_integrate_fire import LeakyIntegrateFire


class LeakyIntegrateFireInput(LeakyIntegrateFire, InputBase):
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, tau_mem: InitValue = 20.0,
                 tau_refrac: InitValue = None, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, scale_i: bool = False,
                 softmax: Optional[bool] = None, input_frames=1, input_frame_time=1):
        super(LeakyIntegrateFireInput, self).__init__(
            v_thresh=v_thresh, v_reset=v_reset, v=v, tau_mem=tau_mem, 
            tau_refrac=tau_refrac, relative_reset=relative_reset, 
            integrate_during_refrac=integrate_during_refrac, scale_i=scale_i,
            softmax=softmax, egp_name="Input", input_frames=input_frames, 
            input_frame_time=input_frame_time)


    def get_model(self, population, dt, batch_size):
        # Get standard integrate-and-fire model
        nm = super(LeakyIntegrateFireInput, self).get_model(population, dt,
                                                            batch_size)

        # Replace standard Isyn input with reference
        # to local variable, add input logic and return
        nm.replace_input("input")
        self.add_input_logic(nm, batch_size, population.shape)
        return nm
