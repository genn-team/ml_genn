import numpy as np

from numbers import Number
from typing import Sequence

class BatchFilter:
    def __init__(self, f):
        # If filter is set to a single boolean value, use it directly as mask
        if isinstance(f, bool):
            self._mask = f
        # Otherwise, if filter is specified with array
        elif isinstance(f, (np.ndarray, Sequence)):
            # Convert to numpy
            array = np.asarray(f)
            
            # If array's dataype is boolean
            # **CHECK** numpy seems to figure this out from lists of 
            # True and False but I can't see this documented anywhere
            if np.issubdtype(array.dtype, bool):
                self._mask =  array
            # Otherwise, if it's integer-derived
            elif np.issubdtype(array.dtype, np.integer):
                # Create suitable mask
                self._mask = np.zeros(np.amax(array) + 1, dtype=bool)
                self._mask[array] = True
            else:
                raise RuntimeError("Unsupported object in bacth filter")
        else:
            raise RuntimeError("Unsupported batch filter format")

    def __getitem__(self, key):
        if isinstance(self._mask, bool):
            return self._mask
        elif key < len(self._mask):
            return self._mask[key]
        else:
            return False
    
class NeuronFilter:
    def __init__(self, f, shape):
        # Normalize shape
        shape = shape if isinstance(shape, Sequence) else (shape,)
        
        # If filter is set to a single boolean value, use it directly as mask
        if isinstance(f, bool):
            self._mask = f            
        # Otherwise, if filter is specified as a slice, create   
        # mask and set elements specified by slice to True
        elif isinstance(f, slice):
            if len(shape) != 1:
                raise RuntimeError("Neuron filters can only be specified "
                                   "as a slice if their shape is 1D")
            self._mask = np.zeros(shape, dtype=bool)
            self._mask[f] = True
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
                self._mask =  array
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
                self._mask = np.zeros(shape, dtype=bool)
                
                # Set elements specified by slices or indices to True
                if len(shape) == 1:
                    self._mask[array] = True
                else:
                    self._mask[tuple(array)] = True
        else:
            raise RuntimeError("Unsupported neuronfilter format")
           
    def __getitem__(self, key):
        if isinstance(self._mask, bool):
            return self._mask
        else:
            return self._mask[key]
