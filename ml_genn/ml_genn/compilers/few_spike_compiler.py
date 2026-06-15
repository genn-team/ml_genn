import numpy as np

from collections import deque, namedtuple
from pygenn import SynapseMatrixType
from typing import Iterator, Optional, Sequence
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from .. import Connection, Population, Network
from ..communicators import Communicator
from ..metrics import Metric
from ..neurons import FewSpikeRelu, FewSpikeReluInput
from ..readouts import Var
from ..synapses import Delta
from ..utils.callback_list import CallbackList
from ..utils.model import NeuronModel, SynapseModel
                           
from ..utils.data import get_dataset_size
from ..utils.module import get_object_mapping
from ..utils.network import get_network_dag, get_underlying_pop
from ..utils.value import is_value_constant

from ..metrics import default_metrics


class CompiledFewSpikeNetwork(CompiledNetwork):
    """Compiled network used for performing inference using
    ANNs converted to SNN using FewSpike encoding [Stockl2021]_.
    """
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, communicator,
                 k: int, pop_pipeline_depth: dict):
        super(CompiledFewSpikeNetwork, self).__init__(
              genn_model, neuron_populations, connection_populations,
              communicator, k)

        self.k = k
        self.pop_pipeline_depth = pop_pipeline_depth

    def evaluate(self, x: dict, y: dict,
                 metrics="sparse_categorical_accuracy",
                 callbacks=[BatchProgressBar()]):
        """ Evaluate an input in numpy format against labels

        Args:
            x:          Dictionary of inputs to inject 
                        into input neuron populations.
            y:          Dictionary of labels to compare to
                        readout from output neuron population.
            metrics:    Metrics to calculate.
            callbacks:  List of callbacks to run during evaluation.
        """
        # Determine the number of elements in x and y
        x_size = get_dataset_size(x)
        y_size = get_dataset_size(y)

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
        splits = range(0, x_size, self.genn_model.batch_size)
        x_batched = [[d[s:s + self.genn_model.batch_size] for s in splits]
                     for d in x.values()]
        y_batched = [[d[s:s + self.genn_model.batch_size] for s in splits] 
                     for d in y.values()]

        # Zip together and evaluate using iterator
        return self.evaluate_batch_iter(list(x.keys()), list(y.keys()),
                                        iter(zip(*(x_batched + y_batched))),
                                        len(splits), metrics, callbacks)

    def evaluate_batch_iter(self, inputs, outputs, data: Iterator,
                            num_batches: Optional[int] = None,
                            metrics="sparse_categorical_accuracy",
                            callbacks=[BatchProgressBar()]):
        """ Evaluate an input in iterator format against labels
        Args:
            x:          Dictionary of inputs to inject 
                        into input neuron populations.
            y:          Dictionary of labels to compare to
                        readout from output neuron population.
            metrics:    Metrics to calculate.
            callbacks:  List of callbacks to run during evaluation.
        """
        # Convert inputs and outputs to tuples
        inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
        outputs = outputs if isinstance(outputs, Sequence) else (outputs,)

        # Build metrics
        metrics = get_object_mapping(metrics, outputs, Metric, 
                                     "Metric", default_metrics)

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
                    batch_y_pred = self.get_readout(o)

                    # Update metrics
                    metrics[o].update(batch_y_true,
                                      batch_y_pred[:len(batch_y_true)],
                                      self.communicator)

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
CompileState = namedtuple("CompileState",
                          ["con_delay", "pop_pipeline_depth"])


class FewSpikeCompiler(Compiler):
    def __init__(self, k: int = 10, dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False,
                 prefer_in_memory_connect: bool = True,
                 communicator: Communicator = None, **genn_kwargs):
        # Determine matrix type order of preference based on flag
        if prefer_in_memory_connect:
            supported_matrix_type = [SynapseMatrixType.SPARSE,
                                     SynapseMatrixType.DENSE,
                                     SynapseMatrixType.TOEPLITZ,
                                     SynapseMatrixType.PROCEDURAL_KERNELG,
                                     SynapseMatrixType.PROCEDURAL]
        else:
            supported_matrix_type = [SynapseMatrixType.TOEPLITZ,
                                     SynapseMatrixType.PROCEDURAL_KERNELG,
                                     SynapseMatrixType.PROCEDURAL,
                                     SynapseMatrixType.SPARSE,
                                     SynapseMatrixType.DENSE]
        super(FewSpikeCompiler, self).__init__(supported_matrix_type, dt,
                                               batch_size, rng_seed,
                                               kernel_profiling, communicator,
                                               **genn_kwargs)
        self.k = k

    def pre_compile(self, network: Network, genn_model, 
                    inputs, outputs, **kwargs) -> CompileState:
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

        return CompileState(con_delay=con_delay,
                            pop_pipeline_depth=pop_pipeline_depth)
    
    def apply_delay(self, genn_pop, conn: Connection,
                    delay, compile_state):
        # Check that no delay is already set
        assert is_value_constant(delay) and delay == 0
        
        # Use pre-calculated delay as axonal
        genn_pop.axonal_delay_steps = compile_state.con_delay[conn]

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state: CompileState) -> NeuronModel:
        # Check neuron model is supported
        if not isinstance(pop.neuron, (FewSpikeRelu, FewSpikeReluInput)):
            raise NotImplementedError(
                "FewSpike models only support FewSpikeRelu "
                "and FewSpikeReluInput neurons")

        # If population has a readout i.e. it's an output
        if pop.neuron.readout is not None:
            # Check readout is supported
            if not isinstance(pop.neuron.readout, Var):
                raise NotImplementedError(
                    "FewSpike models only support output "
                    "neurons with Var readout")

            # Add readout logic to model
            model = pop.neuron.readout.add_readout_logic(model)

        # Build neuron model
        return super(FewSpikeCompiler, self).build_neuron_model(
            pop, model, compile_state)

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state: CompileState) -> SynapseModel:
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("FewSpike models only "
                                      "support Delta synapses")

        return super(FewSpikeCompiler, self).build_synapse_model(
            conn, model, compile_state)

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict, 
                                compile_state: CompileState) -> CompiledFewSpikeNetwork:
        return CompiledFewSpikeNetwork(genn_model, neuron_populations,
                                       connection_populations,
                                       self.communicator, self.k,
                                       compile_state.pop_pipeline_depth)
