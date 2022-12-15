import logging
import numpy as np

from typing import Sequence, Union

logger = logging.getLogger(__name__)

ExampleFilterType = Union[Sequence, np.ndarray, np.integer, int, None]
NeuronFilterType = Union[Sequence, slice, np.ndarray, np.integer, int, None]


class ExampleFilter:
    def __init__(self, f: ExampleFilterType):
        # If no filter is specified
        if f is None:
            self._mask = None
        # Otherwise, if filter is specified with array
        elif isinstance(f, (np.ndarray, Sequence)):
            # Convert to numpy
            array = np.asarray(f)

            # If array's dataype is boolean
            # **CHECK** numpy seems to figure this out from lists of
            # True and False but I can't see this documented anywhere
            if np.issubdtype(array.dtype, bool):
                self._mask = array
            # Otherwise, if it's integer-derived
            elif np.issubdtype(array.dtype, np.integer):
                # Give warning if array isn't ordered
                if not np.all(array[:-1] <= array[1:]):
                    logger.warn("Example filters specified as arrays "
                                "of integers should be sorted otherwise "
                                "recordings will be returned in "
                                "different order")

                # Create suitable mask
                self._mask = np.zeros(np.amax(array) + 1, dtype=bool)
                self._mask[array] = True
            else:
                raise RuntimeError("Unsupported object in example filter")
        elif isinstance(f, (int, np.integer)):
            self._mask = np.zeros(f + 1, dtype=bool)
            self._mask[f] = True
        else:
            raise RuntimeError("Unsupported example filter format")

    def get_batch_mask(self, batch: int, batch_size: int) -> np.ndarray:
        # If no mask is specified, return array of Trues
        if self._mask is None:
            return np.ones(batch_size, dtype=bool)
        else:
            # Get indices of examples at beginning and end of batch
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size

            # Extract slice of mask
            mask_slice = self._mask[batch_start:batch_end]

            # Pad with False out to full batch size
            return np.pad(mask_slice, (0, batch_size - len(mask_slice)),
                          constant_values=False)


def get_neuron_filter_mask(f: NeuronFilterType, shape) -> np.ndarray:
    # Normalize shape
    shape = shape if isinstance(shape, Sequence) else (shape,)

    # If no filter is specified, create mask of ones
    if f is None:
        mask = np.ones(shape, dtype=bool)
    # Otherwise, if filter is specified as a slice, create
    # mask and set elements specified by slice to True
    elif isinstance(f, slice):
        if len(shape) != 1:
            raise RuntimeError("Neuron filters can only be specified "
                               "as a slice if their shape is 1D")
        mask = np.zeros(shape, dtype=bool)
        mask[f] = True
    # Otherwise, if filter is specified as a sequence
    elif isinstance(f, (np.ndarray, Sequence)):
        # Convert to numpy
        array = np.asarray(f)

        # If array's dataype is boolean
        # **CHECK** numpy seems to figure this out from lists of
        # True and False but I can't see this documented anywhere
        if np.issubdtype(array.dtype, bool):
            if array.shape != shape:
                raise RuntimeError(f"Neuron filters specified as arrays "
                                   f"of booleans must have a "
                                   f"shape of {shape}")
            mask = array
        # Otherwise
        else:
            # If array contains objects, check they're all slices
            if np.issubdtype(array.dtype, np.object_):
                if any(not isinstance(o, slice) for o in array):
                    raise RuntimeError("Unsupported object in neuron filter")
            # Otherwise, if array contents aren't integers, give error
            elif not np.issubdtype(array.dtype, np.integer):
                raise RuntimeError("Unsupported neuron filter format")

            # Create suitable mask
            mask = np.zeros(shape, dtype=bool)

            # Set elements specified by slices or indices to True
            # **NOTE** if multi-dimensional, axes container needs to be tuple
            if len(shape) == 1:
                mask[array] = True
            else:
                mask[tuple(array)] = True
    else:
        raise RuntimeError("Unsupported neuron filter format")

    # Return flattened mask - actual GeNN neuron IDs are 1D
    return mask.flatten()
