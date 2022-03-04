from numbers import Number

from . import Initializer

class Uniform(Initializer):
    def __init__(self, min: float=0.0, max: float=1.0):
        super(Uniform, self).__init__()

        self.min = min
        self.max = max
    
        if not(isinstance(self.min, Number)):
            raise RuntimeError("'min' parameter must be a number")
        if not(isinstance(self.max, Number)):
            raise RuntimeError("'max' parameter must be a number")
    
    @property
    def snippet(self):
        return "Uniform"
    
    @property
    def param_vals(self):
        return {"min": self.min, "max": self.max}
    
    def __repr__(self):
        return f"(Uniform) Min: {self.min}, Max: {self.max}"
