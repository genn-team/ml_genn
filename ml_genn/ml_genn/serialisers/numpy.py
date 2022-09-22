from os import path

from .serialiser import Serialiser

class Numpy(Serialiser):
    def __init__(self, path: str = ""):
        self.path = path

    def serialise(self, keys, data):
        np.save(f"{self._get_filename(keys)}.npy", data)

    def deserialise(self, keys):
        return np.load(f"{self._get_filename(keys)}.npy")

    def _get_filename(self, keys):
        return path.join(self.path,
                         "_".join(str(x) for x in keys))