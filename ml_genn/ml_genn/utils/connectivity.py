import numpy as np

from enum import Enum
from ..initializers import Initializer
from ..utils.snippet import InitializerSnippet


class PadMode(Enum):
    VALID = 'valid'
    SAME = 'same'


class KernelInit(Initializer):
    def __init__(self, kernel):
        super(KernelInit, self).__init__()

        self.kernel = kernel

    def get_snippet(self):
        return InitializerSnippet("Kernel", {}, {"kernel": self.kernel})


def get_conv_same_padding(in_size, conv_size, stride):
    # Calculate padding following approach described at
    # https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/nn_ops.py#L48-L88
    if (in_size % stride == 0):
        return max(conv_size - stride, 0) // 2
    else:
        return max(conv_size - (in_size % stride), 0) // 2


def get_param_2d(name, param, default=None):
    if param is None:
        if default is not None:
            return default
        else:
            raise ValueError(f"{name}: cannot be None")

    elif isinstance(param, (list, tuple)):
        if len(param) == 2:
            return tuple(param)
        else:
            raise ValueError(f"{name}: incorrect length: {len(param)}")

    elif isinstance(param, int):
        return (param, param)

    else:
        raise TypeError(f"{name}: incorrect type: {type(param)}")

def update_target_shape(target, output_shape: tuple, flatten_out: bool):
    # If no target shape is set, set to either flattened
    # or un-modified output shape depending on flag
    flat_output_shape = (np.prod(output_shape),)
    if target.shape is None:
        target.shape = (flat_output_shape if flatten_out
                        else output_shape)
    # Otherwise, if output should be flattened and target
    # shape doesn't match flattened output shape, give error
    elif flatten_out and flat_output_shape != target.shape:
        raise RuntimeError(f"Target layer shape {target.shape} doesn't match "
                           f"flattened output shape {flat_output_shape}")
    # Otherwise, if output shouldn't be flattened and target
    # shape doesn't match original output shape, give error
    elif not flatten_out and output_shape != target.shape:
        raise RuntimeError(f"Target layer shape {target.shape} doesn't "
                           f"match output shape {output_shape}")
