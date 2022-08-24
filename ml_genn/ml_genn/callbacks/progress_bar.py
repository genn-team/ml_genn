from .callback import Callback


class BatchProgressBar(Callback):
    def __init__(self):
        self._num_batches = None
        self._progress_bar = None

    def set_params(self, num_batches, **kwargs):
        self._num_batches = num_batches

    def on_batch_end(self, batch, metrics):
        self._display_metrics(metrics)
        self._progress_bar.update(1)

    def on_test_begin(self):
        # Initialise progress bar if required
        self._init_prog_bar()

    def on_test_end(self, metrics):
        self._display_metrics(metrics)
        self._progress_bar.close()

    def _init_prog_bar(self):
        # If there's no existing progress bar,
        if self._progress_bar is None:
            from tqdm import tqdm
            self._progress_bar = tqdm(total=self._num_batches)

        # Reset progress bar
        self._progress_bar.reset()

    def _display_metrics(self, metrics):
        self._progress_bar.set_postfix_str(
            ",".join(f"{type(m).__name__}: {m.result:.4f}"
                     for m in metrics.values()
                     if m.result is not None))
