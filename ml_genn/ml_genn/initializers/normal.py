from .initializer import Initializer
from ..utils.snippet import ConstantValueDescriptor, InitializerSnippet


class Normal(Initializer):
    mean = ConstantValueDescriptor()
    sd = ConstantValueDescriptor()

    def __init__(self, mean: float = 0.0, sd: float = 1.0):
        super(Normal, self).__init__()

        self.mean = mean
        self.sd = sd

    def get_snippet(self):
        return InitializerSnippet("Normal",
                                  {"mean": self.mean, "sd": self.sd})

    def __repr__(self):
        return f"(Normal) Mean: {self.mean}, Standard Deviation: {self.sd}"
