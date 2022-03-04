from re import compile
from inspect import isclass

# Curtesy of https://stackoverflow.com/a/1176023/1476754
camel_to_snake_pattern = compile(r"(?<!^)(?=[A-Z])")

def get_module_models(module, base_class):
    # Loop through names of objects in module
    target_dict = {}
    for name in dir(module):
        # If object is a class derived from
        # base class,but not base class itself
        obj = getattr(module, name)
        if (isclass(obj) and issubclass(obj, base_class) 
                and obj != base_class):
            # Convert its name to snake_cast
            snake_name = camel_to_snake_pattern.sub("_", name).lower()
            target_dict[snake_name] = obj()
    return target_dict

def get_model(model, base_class, description, dictionary):
    if isinstance(model, base_class):
        return model
    elif isinstance(model, str):
        if model in dictionary:
            return dictionary[model]
        else:
            raise RuntimeError(f"{description} model '{neuron}' unknown")
    else:
        raise RuntimeError(f"{description} models should be specified "
                            "either as a string or a {description} object")