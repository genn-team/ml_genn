from typing import Any
from .callback import Callback

from dataclasses import dataclass

@dataclass
class State:
    num_batches: int
    progress_bar: Any = None

class BatchProgressBar(Callback):
    """Callback to display a `tqdm <https://tqdm.github.io/>`_ progress bar
    which gets updated every batch."""
    def __init__(self):

    def create_state(self, num_batches, **kwargs):
        return State(num_batches)

    def on_batch_end(self, state, batch, metric_state):
        self._display_metrics(state, metric_state)
        state.progress_bar.update(1)

    def on_test_begin(self, state):
        self._init_prog_bar(state)

    def on_test_end(self, state, metric_state):
        self._close_prog_bar(state, metric_state)

    def on_train_begin(self, state):
        self._init_prog_bar(state)

    def on_train_end(self, state, metric_state):
        self._close_prog_bar(state, metric_state)

    def on_epoch_begin(self, state, epoch):
        # Set description with epoch and reset so batch returns to 0
        state.progress_bar.set_description(f"Epoch {epoch}")
        state.progress_bar.reset()

    def _init_prog_bar(self, state):
        from tqdm.auto import tqdm
        
        assert state.progress_bar is None
        state.progress_bar = tqdm(total=state.num_batches)

        # Reset progress bar
        state.progress_bar.reset()

    def _close_prog_bar(self, state, metric_state):
        self._display_metrics(state, metric_state)
        state.progress_bar.close()

    def _display_metrics(self, state, metric_state):
        state.progress_bar.set_postfix_str(
            ",".join(f"{type(m).__name__}: {m.result:.4f}"
                     for m in metric_state.values()
                     if m.result is not None))


    
