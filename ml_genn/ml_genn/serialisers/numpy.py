from os import path

from .serialiser import Serialiser

class Numpy(Serialiser):
    def __init__(self, path: str = ""):
        self.path = path

    def serialise(self, keys, data):
        np.save(self._get_filename(keys), data)

    def deserialise(self, keys):
        return np.load(self._get_filename(keys))

    def _get_filename(self, keys):
        return path.join(self.path,
                         "_".join(str(x) for x in keys))