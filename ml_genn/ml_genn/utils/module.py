from inspect import Parameter
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from copy import copy
from re import compile
from inspect import isclass, signature

# Curtesy of https://stackoverflow.com/a/1176023/1476754
camel_to_snake_pattern = compile(r"(?<!^)(?=[A-Z])")


def get_module_classes(globals: Mapping[str, Any],
                       base_class: Type) -> Dict[str, Any]:
    # Loop through names of objects in module
    target_dict = {}
    for name, obj in globals.items():
        # If object is a class derived from
        # base class, but not base class itself
        if (isclass(obj) and issubclass(obj, base_class)
                and obj != base_class):
            # Inspect class constructor to get parameters
            ctr_params = signature(obj.__init__).parameters
  
            # If all of the parameters (aside from self) have a default
            # value or are variable args, class is default constructable
            default_constructable = all((p.kind == Parameter.VAR_POSITIONAL
                                         or p.kind == Parameter.VAR_KEYWORD
                                         or p.default is not Parameter.empty)
                                        for n, p in ctr_params.items()
                                        if n != "self")
            # If this is true, convert class's name
            # to snake_cast and add to dictionary
            if default_constructable:
                snake_name = camel_to_snake_pattern.sub("_", name).lower()
                target_dict[snake_name] = obj
    return target_dict

T = TypeVar("T")

def get_object(obj, base_class: Type[T], description: str,
               dictionary: Mapping[str, T]) -> Optional[T]:
    if obj is None:
        return obj
    elif isinstance(obj, str):
        if obj in dictionary:
            return dictionary[obj]()
        else:
            raise RuntimeError(f"{description} object '{obj}' unknown")
    elif isinstance(obj, base_class):
        return copy(obj)
    else:
        raise RuntimeError(f"{description} objects should be specified "
                           f"either as a string or a {description} object")


def get_object_mapping(obj, keys, base_class: Type,
                       description: str, dictionary) -> dict:
    # If metrics are already in dictionary form
    if isinstance(obj, dict):
        # If keys match, process each metric and create new dictionary
        if set(obj.keys()) == set(keys):
            return {k: get_object(o, base_class, description, dictionary) 
                    for k, o in obj.items()}
        else:
            raise RuntimeError(f"{description} dictionary keys "
                               f"should match network outputs")
    # Otherwise, create new dictionay with
    # copy of processed metric for each output
    else:
        return {k: get_object(obj, base_class, description, dictionary)
                for k in keys}
