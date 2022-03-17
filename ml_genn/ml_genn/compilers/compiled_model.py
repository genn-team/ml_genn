from typing import Sequence, Union
from ..population import Population
from ..layer import InputLayer, Layer

def _get_underlying_pop(obj):
    # Get underling population from object
    if isinstance(obj, Population):
        return obj
    elif isinstance(obj, (InputLayer, Layer)):
        return obj.population()
    else:
        raise RuntimeError(f"{obj} is not a valid Population, "
                           f"InputLayer or Layer object")

class CompiledModel:
    _context = None

    def __init__(self, genn_model, neuron_populations, connection_populations):
        self.genn_model = genn_model
        self.neuron_populations = neuron_populations
        self.connection_populations = connection_populations

    def set_input(self, inputs: dict):
        # Loop through populations
        for pop, input in inputs.items():
            # Find corresponding GeNN population and set input
            pop = _get_underlying_pop(pop)
            pop.neuron.set_input(self.neuron_populations[pop], 
                                 pop.shape, input)
    
    def get_output(self, outputs: Union[Sequence, Population, 
                                        InputLayer, Layer]):
        if isinstance(outputs, Sequence):
            return [self._get_output(p) for p in outputs]
        else:
            return self._get_output(outputs)

    def step_time(self):
        """Step the GeNN model
        """
        self.genn_model.step_time()

    def reset_time(self):
        """Reset the GeNN model"""
        self.genn_model.timestep = 0
        self.genn_model.t = 0.0

    def __enter__(self):
        if CompiledModel._context is not None:
            raise RuntimeError("Nested compiled models are "
                               "not currently supported")

        CompiledModel._context = self
        self.genn_model.build()
        self.genn_model.load()

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        assert CompiledModel._context is not None
        CompiledModel._context = None
    
    def _get_output(self, pop):
        pop = _get_underlying_pop(pop)
        return pop.neuron.get_output(self.neuron_populations[pop], pop.shape)
"""

        self.name = name
        self.layers = []
        self.inputs = inputs
        self.outputs = outputs
        self.g_model = None

        # Construct topologically sorted list of layers (Kahn's algorithm as described here: https://en.wikipedia.org/wiki/Topological_sorting)
        new_layers = set(inputs)
        seen_synapses = set()
        while new_layers:
            layer = new_layers.pop()
            self.layers.append(layer)

            # Explore downstream layers whose upstream synapses have all been seen
            for downstream_synapse in layer.downstream_synapses:
                seen_synapses.add(downstream_synapse)
                if seen_synapses.issuperset(downstream_synapse.target().upstream_synapses):
                    new_layers.add(downstream_synapse.target())

        # Check that output layers are reachable from input layers
        if not all(output in self.layers for output in self.outputs):
            raise ValueError('output layers unreachable from input layers')


        # Input sanity check
        save_samples = list(set(save_samples))
        if any(i < 0 or i >= n_samples for i in save_samples):
            raise ValueError('one or more invalid save_samples value')

        n_correct = [0] * len(self.outputs)
        accuracy = [0] * len(self.outputs)
        all_spikes = [[[] for i,_ in enumerate(self.layers)] for s in save_samples]

        # Pad number of samples so pipeline can be flushed
        pipeline_depth = self.calc_pipeline_depth()
        padded_n_samples = n_samples + (pipeline_depth * self.g_model.batch_size)

        batch_labels_queue = deque()

        # Process batches
        progress = tqdm(total=n_samples)
        for batch_start in range(0, padded_n_samples, self.g_model.batch_size):

            # If any elements of this batch have data (rather than being entirely pipeline padding)
            if batch_start < n_samples:
                batch_end = min(batch_start + self.g_model.batch_size, n_samples)
                batch_data, batch_labels = next(data_iterator)
                batch_data = [batch_data]
                batch_labels = [batch_labels]
                batch_labels_queue.append(batch_labels)

                assert(batch_data[0].shape[0] == batch_end - batch_start)
                assert(batch_labels[0].shape[0] == batch_end - batch_start)

                save_samples_in_batch = [i for i in save_samples if batch_start <= i < batch_end]

                # Set new input
                self.set_input_batch(batch_data)

            # Reset timesteps etc
            self.reset()

            # Main simulation loop
            while self.g_model.t < time:

                # Step time
                self.step_time()

                # Save spikes
                for i in save_samples_in_batch:
                    k = save_samples.index(i)
                    batch_i = i - batch_start
                    for l, layer in enumerate(self.layers):
                        nrn = layer.neurons.nrn
                        nrn.pull_current_spikes_from_device()
                        all_spikes[k][l].append(np.copy(
                            nrn.current_spikes[batch_i] if self.g_model.batch_size > 1
                            else nrn.current_spikes))

            # If first input in batch has passed through
            if batch_start >= (pipeline_depth * self.g_model.batch_size):
                pipe_batch_start = batch_start - (pipeline_depth * self.g_model.batch_size)
                pipe_batch_end = min(pipe_batch_start + self.g_model.batch_size, n_samples)
                batch_labels = batch_labels_queue.popleft()

                # Compute accuracy
                for output_i in range(len(self.outputs)):
                    predictions = self.outputs[output_i].neurons.get_predictions(
                        pipe_batch_end - pipe_batch_start)
                    if batch_labels[output_i].shape != predictions.shape:
                        batch_labels[output_i] = [np.argmax(i) for i in batch_labels[output_i]]
                    n_correct[output_i] += np.sum(predictions == batch_labels[output_i])
                    accuracy[output_i] = (n_correct[output_i] / pipe_batch_end) * 100

                progress.set_postfix_str('accuracy: {:2.2f}'.format(np.mean(accuracy)))
                progress.update(pipe_batch_end - pipe_batch_start)

        progress.close()

        # Create spike index and time lists
        spike_i = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        spike_t = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        for i in range(len(save_samples)):
            for j in range(len(self.layers)):
                spikes = all_spikes[i][j]
                spike_i[i][j] = np.concatenate(spikes)
                spike_t[i][j] = np.concatenate([np.ones_like(s) * i * self.g_model.dT for i, s in enumerate(spikes)])

        return accuracy, spike_i, spike_t


    def evaluate(self, data, labels, time, save_samples=[]):

        # Input sanity check
        n_samples = data[0].shape[0]
        save_samples = list(set(save_samples))
        if len(data) != len(self.inputs):
            raise ValueError('data list length and input layer list length mismatch')
        if len(labels) != len(self.outputs):
            raise ValueError('label list length and output layer list length mismatch')
        if not all(x.shape[0] == n_samples for x in data + labels):
            raise ValueError('sample count mismatch in data and labels arrays')
        if any(i < 0 or i >= n_samples for i in save_samples):
            raise ValueError('one or more invalid save_samples value')

        n_correct = [0] * len(self.outputs)
        accuracy = [0] * len(self.outputs)
        all_spikes = [[[] for i,_ in enumerate(self.layers)] for s in save_samples]

        # Pad number of samples so pipeline can be flushed
        pipeline_depth = self.calc_pipeline_depth()
        padded_n_samples = n_samples + (pipeline_depth * self.g_model.batch_size)

        # Process batches
        progress = tqdm(total=n_samples)
        for batch_start in range(0, padded_n_samples, self.g_model.batch_size):

            # If any elements of this batch have data (rather than being entirely pipeline padding)
            if batch_start < n_samples:
                batch_end = min(batch_start + self.g_model.batch_size, n_samples)
                batch_data = [x[batch_start:batch_end] for x in data]

                save_samples_in_batch = [i for i in save_samples if batch_start <= i < batch_end]

                # Set new input
                self.set_input_batch(batch_data)

            # Reset timesteps etc
            self.reset()

            # Main simulation loop
            while self.g_model.t < time:

                # Step time
                self.step_time()

                # Save spikes
                for i in save_samples_in_batch:
                    k = save_samples.index(i)
                    batch_i = i - batch_start
                    for l, layer in enumerate(self.layers):
                        nrn = layer.neurons.nrn
                        nrn.pull_current_spikes_from_device()
                        all_spikes[k][l].append(np.copy(
                            nrn.current_spikes[batch_i] if self.g_model.batch_size > 1
                            else nrn.current_spikes))

            # If first input in batch has passed through
            if batch_start >= (pipeline_depth * self.g_model.batch_size):
                pipe_batch_start = batch_start - (pipeline_depth * self.g_model.batch_size)
                pipe_batch_end = min(pipe_batch_start + self.g_model.batch_size, n_samples)
                batch_labels = [y[pipe_batch_start:pipe_batch_end] for y in labels]

                # Compute accuracy
                for output_i in range(len(self.outputs)):
                    predictions = self.outputs[output_i].neurons.get_predictions(
                        pipe_batch_end - pipe_batch_start)
                    if batch_labels[output_i].shape != predictions.shape:
                        batch_labels[output_i] = [np.argmax(i) for i in batch_labels[output_i]]
                    n_correct[output_i] += np.sum(predictions == batch_labels[output_i])
                    accuracy[output_i] = (n_correct[output_i] / pipe_batch_end) * 100

                progress.set_postfix_str('accuracy: {:2.2f}'.format(np.mean(accuracy)))
                progress.update(pipe_batch_end - pipe_batch_start)

        progress.close()

        # Create spike index and time lists
        spike_i = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        spike_t = [[None for i,_ in enumerate(self.layers)] for s in save_samples]
        for i in range(len(save_samples)):
            for j in range(len(self.layers)):
                spikes = all_spikes[i][j]
                spike_i[i][j] = np.concatenate(spikes)
                spike_t[i][j] = np.concatenate([np.ones_like(s) * i * self.g_model.dT for i, s in enumerate(spikes)])

        return accuracy, spike_i, spike_t


    def calc_pipeline_depth(self):
        # If none of the layers have the pipelined attribute, return 0
        if all(not hasattr(l.neurons, "pipelined")
               for l in self.layers if l not in self.outputs):
           return 0
       
        # If there are multiple inputs, give an error
        # **NOTE** inputs would have to be injected at different times to relax this
        if len(self.inputs) > 1:
            raise NotImplementedError("Pipelined models with multiple inputs "
                                      "are not currently supported")
        
        # If there are multiple outputs, give an error
        # **NOTE** outputs would need to be retrieved at different times to relax this
        if len(self.outputs) > 1:
            raise NotImplementedError("Pipelined models with multiple outputs "
                                      "are not currently supported")
        # Recursive function to get delay along (arbitrary) path to target
        def calc_delay(synapse, target):
            # If we've hit target, stop
            layer = synapse.target()
            if layer == target:
                return 0

            # Recurse through first downstream synapse
            return synapse.delay + 1 + calc_delay(layer.downstream_synapses[0], target)
        
        # Calculate delay from input to output
        # **NOTE** in pipelined networks, delay should have been balanced
        return calc_delay(self.inputs[0].downstream_synapses[0], self.outputs[0])

    """
