import ml_genn

from collections import Counter
from typing import Optional
from varname import (ImproperUseError, MultiTargetAssignmentWarning,
                     VarnameRetrievingError)

from varname import varname

def _get_varname() -> str:
    # Try and get first frame assigning varname outside of mlGeNN
    try:
        return varname(frame=0, ignore=ml_genn)
    except (ImproperUseError, MultiTargetAssignmentWarning,
            VarnameRetrievingError):
        return None
    
class UniqueName:
    def __init__(self):
        self._counter = Counter()
    
    def __call__(self, name: Optional[str], auto_name: str):
        # If name isn't provided, try and extract name of assigned variable
        name = name or _get_varname()

        # If that fails, use automatic name provided by mlGeNN
        name = name or auto_name
        assert name is not None

        # Determine how many times this name has been used already
        name_count = self._counter[name]

        # Increment counter
        self._counter[name] += 1

        # If this isn't the first usage, add suffix
        if name_count > 0:
            name = f"{name}_{name_count}"
        
        return name

    