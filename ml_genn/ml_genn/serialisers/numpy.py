import logging
import os
import numpy as np
from glob import glob

from .serialiser import Serialiser

logger = logging.getLogger(__name__)

class Numpy(Serialiser):
    """Basic numpy based serialiser. Stores arrays as individual numpy binary
    files with paths mapped to filesystem directory structure.
    
    Args:
        path:   File system path to serialise and deserialise relative to
    """
    def __init__(self, path: str = ""):
        self.path = path
        
        # If a path is specified which doesn't exist, create it
        if self.path != "" and not os.path.exists(self.path):
            os.makedirs(self.path)

    def serialise(self, keys, data):
        np.save(self._get_filename(keys, ".npy"), data)

    def deserialise(self, keys):
        return np.load(self._get_filename(keys, ".npy"))
    
    def deserialise_all(self, keys):
        title = self._get_file_title(keys)
        logger.info(f"Deserialising {title}")

        # Loop through files under keys
        data = {}
        for f in glob(self._get_filename(keys, "-*.npy")):
            # Extract file title without path or extension from filename
            title = os.path.splitext(os.path.split(f)[1])[0]
            logger.info(f"\tLoading from {f}")

            # Split title into keys and slice out those that we seperated
            file_keys = title.split("-")[len(keys):]

            # Add file to dictionary with this key
            assert len(file_keys) == 1
            data[file_keys[0]] = np.load(f)
        return data

    def _get_file_title(self, keys):
        return "-".join(str(x) for x in keys)

    def _get_filename(self, keys, suffix):
        return os.path.join(self.path,
                            self._get_file_title(keys) + suffix)
