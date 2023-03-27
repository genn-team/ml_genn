from collections import namedtuple
from numbers import Number

ConnectivitySnippet = namedtuple("ConnectivitySnippet",
                                 ["snippet", "matrix_type", "weight", "delay",
                                  "pre_ind", "post_ind", "trainable"],
                                 defaults=[None, None, True])

InitializerSnippet = namedtuple("InitializerSnippet",
                                ["snippet", "param_vals", "egp_vals"],
                                defaults=[{}, {}])


class ConstantValueDescriptor:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        # If descriptor is accessed as a class attribute, return it
        if instance is None:
            return self
        # Otherwise, return attribute value
        else:
            return getattr(instance, f"_{self.name}")

    def __set__(self, instance, value):
        if isinstance(value, Number):
            setattr(instance, f"_{self.name}", value)
        else:
            raise RuntimeError(f"{self.name} initializers should "
                               f"be specified as numbers")
