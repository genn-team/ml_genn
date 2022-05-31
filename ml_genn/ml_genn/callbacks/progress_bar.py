class BatchProgressBar:
    def __init__(self):
        self._num_batches = None
        self._progress_bar = None

    def set_params(self, params):
        # If number of batches is passes as a parameter, store
        if "num_batches" in params:
            self._num_batches = params["num_batches"]

    def on_batch_end(self, metrics):
        self._progress_bar.update(1)
    
    def on_test_begin(self):
        # Initialise progress bar if required
        self._init_prog_bar()

    def on_test_end(self, metrics):
        self._progress_bar.close()
    
    def _init_prog_bar(self):
        # If there's no existing progress bar, 
        if self._progress_bar is None:
            from tqdm import tqdm
            self._progress_bar = tqdm(total=self._num_batches)

        # Reset progress bar
        self._progress_bar.reset()
            
        
