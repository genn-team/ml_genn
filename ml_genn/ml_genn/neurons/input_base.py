import numpy as np

class InputBase:
    def __init__(self, var_name="Input", **kwargs):
        super(InputBase, self).__init__(**kwargs)

        self.var_name = var_name

    def set_input(self, genn_pop, shape, input):
        # Ensure input is something convertable to a numpy array
        input = np.asarray(input)
        if input.shape != shape:
            raise RuntimeError(f"Input shape {input.shape} does not match "
                                "population shape {shape}")
        
        # Flatten input, copy into variables and push to device
        genn_pop.vars[self.var_name] = input.flatten()
        genn_pop.push_var_to_device(self.var_name)