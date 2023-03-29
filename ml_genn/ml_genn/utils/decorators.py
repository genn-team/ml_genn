from ..network import Network

from functools import wraps
from inspect import signature

# Based on https://stackoverflow.com/a/58983447
def network_default_params(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Get list of function argument names
        arg_names = list(signature(f).parameters.keys())
        
        # Slice out names of arguments (as opposed to kwargs)
        passed_arg_names = arg_names[:len(args)]
        
        # Get default parameters for type
        default_params = Network.get_default_params(type(args[0]))

        # Update defaults with passed kwargs if argument wasn't provided
        final_kwargs = {
            key: value
            for key, value in {**default_params, **kwargs}.items() 
            if key not in passed_arg_names}
            
        return f(*args, **final_kwargs)
    return wrapper