from numbers import Number

from .initializer import Initializer
from ..utils import InitializerSnippet

class Uniform(Initializer):
    def __init__(self, min: float=0.0, max: float=1.0):
        super(Uniform, self).__init__()

        self.min = min
        self.max = max
    
        if not(isinstance(self.min, Number)):
            raise RuntimeError("'min' parameter must be a number")
        if not(isinstance(self.max, Number)):
            raise RuntimeError("'max' parameter must be a number")
    
    def get_snippet(self):
        return InitializerSnippet("Uniform", 
                                  {"min": self.min, "max": self.max}, {})
    
    def __repr__(self):
        return f"(Uniform) Min: {self.min}, Max: {self.max}"
