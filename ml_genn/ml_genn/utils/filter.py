import numpy as np

from numbers import Number
from typing import Sequence

class Filter:
    def __init__(self, f, count=None):
        # If filter is specified as a slice, create mask  
        # and set elements specified by slice to True
        if isinstance(f, slice):
            if count is None:
                raise RuntimeError("Filters can only be specified as a slice "
                                   "if the number of elements is known")
            self._mask = np.zeros(count, dtype=bool)
            self._mask[f] = True
        # Otherwise, if filter is specified with array
        elif isinstance(f, (np.ndarray, Sequence)):
            # Convert to numpy
            array = np.asarray(f)
            
            # If array's dataype is boolean
            # **CHECK** numpy seems to figure this out from lists of 
            # True and False but I can't see this documented anywhere
            if np.issubdtype(array.dtype, bool):
                if count is not None and array.shape != (count, ):
                    raise RuntimeError(f"Filters specified as arrays "
                                       f"of booleans  must contain "
                                       f"{count} elements")
                self._mask =  array
            # Otherwise, if it's integer-derived
            elif np.issubdtype(array.dtype, np.integer):
                max_value = np.amax(array)
                if np.amin(array) < 0:
                    raise RuntimeError("Filters specified as arrays "
                                       "of integers must not contain "
                                       "values < zero")
                if count is not None and max_value >= count:
                    raise RuntimeError(f"Filters specified as arrays "
                                       f"of integers must not contain "
                                       f"values >= {count}")
                
                # Create suitable mask
                self._mask = np.zeros((max_value + 1 if count is None 
                                       else count), dtype=bool)
                self._mask[array] = True
            
    def __getitem__(self, key):
        if key < 0:
            raise IndexError("Filters can only be indexed "
                             "with positive integers")
        elif key < len(self._mask):
            return self._mask[key]
        else:
            return False
            
            
            