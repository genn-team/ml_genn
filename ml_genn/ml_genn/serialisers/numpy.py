import numpy as np
from glob import glob
from os import path

from .serialiser import Serialiser

class Numpy(Serialiser):
    def __init__(self, path: str = ""):
        self.path = path

    def serialise(self, keys, data):
        np.save(self._get_filename(keys, ".npy"), data)

    def deserialise(self, keys):
        return np.load(self._get_filename(keys, ".npy"))
    
    def deserialise_all(self, keys):
        title = self._get_file_title(keys)
        
        # Loop through files under keys
        data = {}
        for f in glob(self._get_filename(keys, "_*.npy")):
            pass
        return data

    def _get_file_title(self, keys):
        return "_".join(str(x) for x in keys)

    def _get_filename(self, keys, suffix):
        return path.join(self.path,
                         self._get_file_title(keys) + suffix)