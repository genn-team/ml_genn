from collections import defaultdict
from typing import List, Sequence
from warnings import warn

from ..callbacks import Callback

from .module import get_object

from ..callbacks import default_callbacks


def _filter_callbacks(callbacks: Sequence[Callback], 
                      method: str) -> List[Callback]:
    return [c for c in callbacks if hasattr(c, method)]


class CallbackList:
    """Class used internally to efficiently handle lists of callback objects
    """
    def __init__(self, callbacks: Sequence[Callback], **params):
        # Build list of callback objects
        self._callbacks = [get_object(c, Callback, "Callback",
                                      default_callbacks)
                           for c in callbacks]

        # Because callback objects themselves should not be stateful,
        # create dictionary to hold any data callbacks produce
        self._data = {}

        # Loop through callbacks, build dictionary of all callbacks
        # of each type and call set_params methods if present
        callback_types = defaultdict(list)
        for c in self._callbacks:
            callback_types[type(c)].append(c)

            if hasattr(c, "set_params"):
                c.set_params(data=self._data, **params)

        # Loop through all callback types and
        # inform the first callback of each type
        # that it is the first in the list
        for t, c in callback_types.items():
            if hasattr(t, "set_first"):
                c[0].set_first()

        # Get lists of callbacks
        self._on_test_begin_callbacks =\
            _filter_callbacks(self._callbacks, "on_test_begin")
        self._on_test_end_callbacks =\
            _filter_callbacks(self._callbacks, "on_test_end")
        self._on_train_begin_callbacks =\
            _filter_callbacks(self._callbacks, "on_train_begin")
        self._on_train_end_callbacks =\
            _filter_callbacks(self._callbacks, "on_train_end")
        self._on_epoch_begin_callbacks =\
            _filter_callbacks(self._callbacks, "on_epoch_begin")
        self._on_epoch_end_callbacks =\
            _filter_callbacks(self._callbacks, "on_epoch_end")
        self._on_batch_begin_callbacks =\
            _filter_callbacks(self._callbacks, "on_batch_begin")
        self._on_batch_end_callbacks =\
            _filter_callbacks(self._callbacks, "on_batch_end")
        self._on_timestep_begin_callbacks =\
            _filter_callbacks(self._callbacks, "on_timestep_begin")
        self._on_timestep_end_callbacks =\
            _filter_callbacks(self._callbacks, "on_timestep_end")

    def on_test_begin(self):
        for c in self._on_test_begin_callbacks:
            c.on_test_begin()

    def on_test_end(self, metrics):
        for c in self._on_test_end_callbacks:
            c.on_test_end(metrics)

    def on_train_begin(self):
        for c in self._on_train_begin_callbacks:
            c.on_train_begin()

    def on_train_end(self, metrics):
        for c in self._on_train_end_callbacks:
            c.on_train_end(metrics)

    def on_epoch_begin(self, epoch: int):
        for c in self._on_epoch_begin_callbacks:
            c.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, metrics):
        for c in self._on_epoch_end_callbacks:
            c.on_epoch_end(epoch, metrics)

    def on_batch_begin(self, batch: int):
        for c in self._on_batch_begin_callbacks:
            c.on_batch_begin(batch)

    def on_batch_end(self, batch: int, metrics):
        for c in self._on_batch_end_callbacks:
            c.on_batch_end(batch, metrics)

    def on_timestep_begin(self, timestep: int):
        for c in self._on_timestep_begin_callbacks:
            c.on_timestep_begin(timestep)

    def on_timestep_end(self, timestep: int):
        for c in self._on_timestep_end_callbacks:
            c.on_timestep_end(timestep)

    def __getitem__(self, index: int):
        return self._callbacks[index]
    
    def get_data(self):
        # Loop through callbacks
        cb_data = {}
        for i, c in enumerate(self._callbacks):
            # If callback has method to get data
            if hasattr(c, "get_data"):
                # Get data
                key, data = c.get_data()

                # Add data to dictionary, using index of
                # callback as key if key is not provided
                if key is None:
                    warn(f"No key provided - a default numerical key {i} will be used")
                    key = i
                if key in cb_data:
                    raise KeyError(f"Callback data key {key} is not unique")
                cb_data[key] = data

        return cb_data
