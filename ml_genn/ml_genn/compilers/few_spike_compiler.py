import numpy as np

from collections import deque, namedtuple
from typing import Iterator, Optional, Sequence
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..neurons import FewSpikeRelu, FewSpikeReluInput
from ..synapses import Delta
from ..utils.callback_list import CallbackList

from ..utils.data import get_metrics, get_numpy_size
from ..utils.network import get_network_dag, get_underlying_pop
from ..utils.value import is_value_constant


class CompiledFewSpikeNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, k: int, pop_pipeline_depth: dict):
        super(CompiledFewSpikeNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations, k)

        self.k = k
        self.pop_pipeline_depth = pop_pipeline_depth

    def evaluate_numpy(self, x: dict, y: dict,
                       metrics="sparse_categorical_accuracy",
                       callbacks=[BatchProgressBar()]):
        """ Evaluate an input in numpy format against labels
        accuracy --  dictionary containing accuracy of predictions
                     made by each output Population or Layer
        """
        # Determine the number of elements in x and y
        x_size = get_numpy_size(x)
        y_size = get_numpy_size(x)

        if x_size is None:
            raise RuntimeError("Each input population must be "
                               " provided with same number of inputs")
        if y_size is None:
            raise RuntimeError("Each output population must be "
                               " provided with same number of labels")
        if x_size != y_size:
            raise RuntimeError("Number of inputs and labels must match")

        # Batch x and y
        # [[in_0_batch_0, in_0_batch_1], [in_1_batch_1, in_1_batch_1]]
        splits = range(self.genn_model.batch_size, x_size,
                       self.genn_model.batch_size)
        x_batched = [np.split(d, splits, axis=0) for d in x.values()]
        y_batched = [np.split(d, splits, axis=0) for d in y.values()]

        # Zip together and evaluate using iterator
        return self.evaluate_batch_iter(list(x.keys()), list(y.keys()),
                                        iter(zip(*(x_batched + y_batched))),
                                        len(splits), metrics, callbacks)

    def evaluate_batch_iter(self, inputs, outputs, data: Iterator,
                            num_batches: Optional[int] = None,
                            metrics="sparse_categorical_accuracy",
                            callbacks=[BatchProgressBar()]):
        """ Evaluate an input in iterator format against labels
        accuracy --  dictionary containing accuracy of predictions
                     made by each output Population or Layer
        """
        # Convert inputs and outputs to tuples
        inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
        outputs = outputs if isinstance(outputs, Sequence) else (outputs,)

        # Build metrics
        metrics = get_metrics(metrics, outputs)

        # Get the pipeline depth of each output
        y_pipe_depth = {
            o: (self.pop_pipeline_depth[get_underlying_pop(o)]
                if get_underlying_pop(o) in self.pop_pipeline_depth
                else 0)
            for o in outputs}

        # Create callback list and begin testing
        num_batches = (None if num_batches is None
                       else num_batches + 1 + max(y_pipe_depth.values()))
        callback_list = CallbackList(callbacks, compiled_network=self,
                                     num_batches=num_batches)
        callback_list.on_test_begin()

        # Build deque to hold y
        y_pipe_queue = {p: deque(maxlen=d + 1)
                        for p, d in y_pipe_depth.items()}

        # While there is data remaining or any y values left in queues
        data_remaining = True
        batch_i = 0
        while data_remaining or any(len(q) > 0
                                    for q in y_pipe_queue.values()):
            # Attempt to get next batch of data,
            # clear data remaining flag if none remains
            try:
                batch_x, batch_y = next(data)
            except StopIteration:
                data_remaining = False

            # Reset time to 0
            # **YUCK** I don't REALLY like this
            self.genn_model.timestep = 0
            self.genn_model.t = 0.0

            # If there is any data remaining,
            if data_remaining:
                # Set x as input
                # **YUCK** this isn't quite right as batch_x
                # could also have outer dimension
                if len(inputs) == 1:
                    self.set_input({inputs[0]: batch_x})
                else:
                    self.set_input({p: x for p, x in zip(inputs, batch_x)})

                # Add each y to correct queue(s)
                # **YUCK** this isn't quite right as batch_y
                # could also have outer dimension
                if len(outputs) == 1:
                    y_pipe_queue[outputs[0]].append(batch_y)
                else:
                    for p, y in zip(outputs, batch_y):
                        y_pipe_queue[p].append(y)

            # Start batch
            callback_list.on_batch_begin(batch_i)

            # Simulate K timesteps
            for t in range(self.k):
                self.step_time(callback_list)

            # Loop through outputs
            for o in outputs:
                # If there is output to read from this population
                if batch_i >= y_pipe_depth[o] and len(y_pipe_queue[o]) > 0:
                    # Pop correct labels from queue
                    batch_y_true = y_pipe_queue[o].popleft()

                    # Get predictions from model
                    batch_y_pred = self.get_output(o)

                    # Update metrics
                    metrics[o].update(batch_y_true,
                                      batch_y_pred[:len(batch_y_true)])

            # End batch
            callback_list.on_batch_end(batch_i, metrics)

            # Next batch
            batch_i += 1

        # End testing
        callback_list.on_test_end(metrics)

        # Return metrics
        return metrics, callback_list.get_data()


# Because we want the converter class to be reusable, we don't want
# the data to be a member, instead we encapsulate it in a tuple
PreCompileOutput = namedtuple("PreCompileOutput",
                              ["con_delay", "pop_pipeline_depth"])


class FewSpikeCompiler(Compiler):
    def __init__(self, k: int = 10, dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False,
                 prefer_in_memory_connect: bool = True, **genn_kwargs):
        super(FewSpikeCompiler, self).__init__(dt, batch_size, rng_seed,
                                               kernel_profiling,
                                               prefer_in_memory_connect,
                                               **genn_kwargs)
        self.k = k

    def pre_compile(self, network, inputs, outputs, **kwargs):
        dag = get_network_dag(inputs, outputs)

        # Loop through populations
        con_delay = {}
        pop_pipeline_depth = {}
        next_pipeline_depth = {}
        for p in dag:
            # If population has incoming connections
            if len(p.incoming_connections) > 0:
                # Determine the maximum pipeline depth from upstream synapses
                pipeline_depth = max(next_pipeline_depth[c().source()]
                                     for c in p.incoming_connections)
                pop_pipeline_depth[p] = pipeline_depth

                # Downstream layer pipeline depth is one more than this
                next_pipeline_depth[p] = pipeline_depth + 1

                # Loop through incoming connections
                for c in p.incoming_connections:
                    # Set upstream delay so all spikes
                    # arrive at correct  pipeline stage
                    source_pop = c().source()
                    depth_difference = (pipeline_depth 
                                        - next_pipeline_depth[source_pop])
                    con_delay[c()] = depth_difference * self.k

            # Otherwise (layer is an input layer),
            # set this layer's delay as zero
            else:
                pop_pipeline_depth[p] = 0
                next_pipeline_depth[p] = 0

        return PreCompileOutput(con_delay=con_delay,
                                pop_pipeline_depth=pop_pipeline_depth)

    def calculate_delay(self, conn, delay, pre_compile_output):
        # Check that no delay is already set
        assert is_value_constant(delay) and delay == 0

        return pre_compile_output.con_delay[conn]

    def build_neuron_model(self, pop, model, custom_updates,
                           pre_compile_output):
        # Check neuron model is supported
        if not isinstance(pop.neuron, (FewSpikeRelu, FewSpikeReluInput)):
            raise NotImplementedError(
                "FewSpike models only support FewSpikeRelu "
                "and FewSpikeReluInput neurons")

        # Build neuron model
        return super(FewSpikeCompiler, self).build_neuron_model(
            pop, model, custom_updates, pre_compile_output)

    def build_synapse_model(self, conn, model, custom_updates,
                            pre_compile_output):
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("FewSpike models only "
                                      "support Delta synapses")

        return super(FewSpikeCompiler, self).build_synapse_model(
            conn, model, custom_updates, pre_compile_output)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, pre_compile_output):
        return CompiledFewSpikeNetwork(genn_model, neuron_populations,
                                       connection_populations, self.k,
                                       pre_compile_output.pop_pipeline_depth)
