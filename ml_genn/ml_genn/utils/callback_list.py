from typing import Sequence

def _filter_callbacks(callbacks: Sequence, method: str):
    return [c for c in callbacks if hasattr(c, method)]

class CallbackList:
    def __init__(self, callbacks: Sequence, **params):
        self.callbacks = list(callbacks)
        
        # Loop through callbacks and call set_params methods if presetn
        for c in self.callbacks:
            if hasattr(c, "set_params"):
                c.set_params(params)

        # Get lists of callbacks
        self._on_test_begin_callbacks =\
            _filter_callbacks(self.callbacks, "on_test_begin")
        self._on_test_end_callbacks =\
            _filter_callbacks(self.callbacks, "on_test_end")
        self._on_batch_begin_callbacks =\
            _filter_callbacks(self.callbacks, "on_batch_begin")
        self._on_batch_end_callbacks =\
            _filter_callbacks(self.callbacks, "on_batch_end")
        self._on_timestep_begin_callbacks =\
            _filter_callbacks(self.callbacks, "on_timestep_begin")
        self._on_timestep_end_callbacks =\
            _filter_callbacks(self.callbacks, "on_timestep_end")

    def on_test_begin(self):
        for c in self._on_test_begin_callbacks:
            c.on_test_begin()

    def on_test_end(self, metrics):
        for c in self._on_test_end_callbacks:
            c.on_test_end(metrics)

    def on_batch_begin(self):
        for c in self._on_batch_begin_callbacks:
            c.on_batch_begin()

    def on_batch_end(self, metrics):
        for c in self._on_batch_end_callbacks:
            c.on_batch_end(metrics)
        
    def on_timestep_begin(self):
        for c in self._on_timestep_begin_callbacks:
            c.on_timestep_begin()

    def on_timestep_end(self):
        for c in self._on_timestep_end_callbacks:
            c.on_timestep_end()
