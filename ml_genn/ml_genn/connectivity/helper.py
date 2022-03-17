from enum import Enum
from ..initializers import Initializer
from ..utils import InitializerSnippet

class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'


class KernelInit(Initializer):
    def __init__(self, kernel):
        super(KernelInit, self).__init__()
        
        self.kernel = kernel
    
    def get_snippet(self):
        return InitializerSnippet("Kernel", {}, {"kernel": self.kernel})
        
def _get_conv_same_padding(in_size, conv_size, stride):
    # Calculate padding following approach described at
    # https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/nn_ops.py#L48-L88
    if (in_size % stride == 0):
        return max(conv_size - stride, 0) // 2
    else:
        return max(conv_size - (in_size % stride), 0) // 2
    
def _get_param_2d(name, param, default=None):

    if param is None:
        if default is not None:
            return default
        else:
            raise ValueError('{}: cannot be None'.format(name))

    elif isinstance(param, (list, tuple)):
        if len(param) == 2:
            return tuple(param)
        else:
            raise ValueError('{}: incorrect length: {}'.format(name, len(param)))

    elif isinstance(param, int):
        return (param, param)

    else:
        raise TypeError('{}: incorrect type: {}'.format(name, type(param)))
