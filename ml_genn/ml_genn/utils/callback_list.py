from collections import defaultdict
from typing import Any, List, Mapping, Sequence
from warnings import warn

from ..callbacks import Callback

from .module import get_object

from ..callbacks import default_callbacks


def _filter_callbacks(callbacks: Sequence[Callback], 
                      state: Mapping[Callback, Any],
                      method: str) -> List[Callback]:
    return [(c, state.get(c)) for c in callbacks if hasattr(c, method)]


class CallbackList:
    """Class used internally to efficiently handle lists of callback objects
    """
    def __init__(self, callbacks: Sequence[Callback], **params):
        # Build list of callback objects
        self._callbacks = [get_object(c, Callback, "Callback",
                                      default_callbacks)
                           for c in callbacks]

        # Because callback objects themselves should not be stateful,
        # create dictionary to hold any state required by callbacks
        self._state = {}

        # Loop through callbacks, build dictionary of all callbacks
        # of each type and call set_params methods if present
        callback_types = defaultdict(list)
        for c in self._callbacks:
            callback_types[type(c)].append(c)

            if hasattr(c, "create_state"):
                state = c.create_state(**params)
                if state is not None:
                    self._state[c] = state

        # Loop through all callback types and
        # inform the first callback of each type
        # that it is the first in the list
        for t, c in callback_types.items():
            if hasattr(t, "set_first"):
                c[0].set_first(self._state.get(c[0]))

        # Get lists of callbacks
        self._on_test_begin_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_test_begin")
        self._on_test_end_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_test_end")
        self._on_train_begin_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_train_begin")
        self._on_train_end_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_train_end")
        self._on_epoch_begin_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_epoch_begin")
        self._on_epoch_end_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_epoch_end")
        self._on_batch_begin_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_batch_begin")
        self._on_batch_end_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_batch_end")
        self._on_timestep_begin_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_timestep_begin")
        self._on_timestep_end_callbacks =\
            _filter_callbacks(self._callbacks, self._state, "on_timestep_end")

    def on_test_begin(self):
        for c, s in self._on_test_begin_callbacks:
            c.on_test_begin(s)

    def on_test_end(self, metric_state):
        for c, s in self._on_test_end_callbacks:
            c.on_test_end(s, metric_state)

    def on_train_begin(self):
        for c, s in self._on_train_begin_callbacks:
            c.on_train_begin(s)

    def on_train_end(self, metric_state):
        for c, s in self._on_train_end_callbacks:
            c.on_train_end(s, metric_state)

    def on_epoch_begin(self, epoch: int):
        for c, s in self._on_epoch_begin_callbacks:
            c.on_epoch_begin(s, epoch)

    def on_epoch_end(self, epoch: int, metric_state):
        for c, s in self._on_epoch_end_callbacks:
            c.on_epoch_end(s, epoch, metric_state)

    def on_batch_begin(self, batch: int):
        for c, s in self._on_batch_begin_callbacks:
            c.on_batch_begin(s, batch)

    def on_batch_end(self, batch: int, metric_state):
        for c, s in self._on_batch_end_callbacks:
            c.on_batch_end(s, batch, metric_state)

    def on_timestep_begin(self, timestep: int):
        for c, s in self._on_timestep_begin_callbacks:
            c.on_timestep_begin(s, timestep)

    def on_timestep_end(self, timestep: int):
        for c, s in self._on_timestep_end_callbacks:
            c.on_timestep_end(s, timestep)

    def __getitem__(self, index: int):
        return self._callbacks[index]
    
    def get_data(self) -> dict:
        # Loop through callbacks
        cb_data = {}
        for i, c in enumerate(self._callbacks):
            # If callback has method to get data
            if hasattr(c, "get_data"):
                # Get data
                key, data = c.get_data(self._state.get(c))

                # Add data to dictionary, using index of
                # callback as key if key is not provided
                if key is None:
                    warn(f"No key provided - a default numerical key {i} will be used")
                    key = i
                if key in cb_data:
                    raise KeyError(f"Callback data key {key} is not unique")
                cb_data[key] = data

        return cb_data
