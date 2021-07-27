
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
