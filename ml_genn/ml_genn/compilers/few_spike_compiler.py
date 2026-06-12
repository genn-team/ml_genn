import numpy as np

from collections import deque, namedtuple
from pygenn import SynapseMatrixType
from typing import Iterator, Optional, Sequence, Union
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from .. import Connection, Population, Network
from ..communicators import Communicator
from ..metrics import Metric, MetricsType
from ..neurons import FewSpikeRelu, FewSpikeReluInput
from ..readouts import Var
from ..synapses import Delta
from ..utils.callback_list import CallbackList
from ..utils.model import NeuronModel, SynapseModel
                           
from ..utils.data import batch_dataset, get_dataset_size
from ..utils.module import get_object_mapping
from ..utils.network import get_network_dag, get_underlying_pop, PopulationType
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

        self.evaluate_timesteps = k
        self.pop_pipeline_depth = pop_pipeline_depth
    
    def evaluate(self, x: dict, y: dict,
                 metrics: MetricsType = "sparse_categorical_accuracy",
                 callbacks=[BatchProgressBar()]):
        """ Evaluate metrics on a numpy dataset
        
        Args:
            x:          Dictionary of testing inputs
            y:          Dictionary of testing labels to compare 
                        predictions against
            metrics:    Metrics to calculate.
            callbacks:  List of callbacks to run during inference.
        """
        # Determine the number of elements in x and y
        x_size = get_dataset_size(x)
        y_size = get_dataset_size(y)

        # Build metrics
        metrics = get_object_mapping(metrics, y.keys(), Metric, 
                                     "Metric", default_metrics)

        if x_size is None:
            raise RuntimeError("Each input population must be "
                               " provided with same number of inputs")
        if y_size is None:
            raise RuntimeError("Each output population must be "
                               " provided with same number of labels")
        if x_size != y_size:
            raise RuntimeError("Number of inputs and labels must match")

        # Batch x and y
        batch_size = self.genn_model.batch_size
        x = batch_dataset(x, batch_size, x_size)
        y = batch_dataset(y, batch_size, y_size)

        # Create callback list and begin testing
        callback_list = CallbackList(callbacks,
                                     compiled_network=self,
                                     num_batches=len(x))
        callback_list.on_test_begin()

        # Loop through batches and evaluate
        for batch_i, (x_batch, y_batch) in enumerate(zip(x, y)):
            self._evaluate_batch(batch_i, x_batch, y_batch,
                                 metrics, callback_list)

        # End testing
        callback_list.on_test_end(metrics)

        # Return metrics
        return metrics, callback_list.get_data()

    def predict(self, x: dict, outputs: Union[Sequence, PopulationType],
                callbacks=[BatchProgressBar()]):
        """ Generate predictions from a numpy dataset
  
        Args:
            x:          Dictionary of testing inputs
            outputs:    Output population(s) to extract predictions from
            callbacks:  List of callbacks to run during inference.
        """
        # Determine the number of elements in x
        x_size = get_dataset_size(x)

        if x_size is None:
            raise RuntimeError("Each input population must be "
                               " provided with same number of inputs")
        # Batch x
        x = batch_dataset(x, self.genn_model.batch_size, x_size)

        # Convert outputs to sequence
        outputs = outputs if isinstance(outputs, Sequence) else [outputs]

        # Create callback list and begin testing
        callback_list = CallbackList(self.base_callbacks + callbacks,
                                     compiled_network=self,
                                     num_batches=len(x))
        callback_list.on_test_begin()

        # Build dictionary mapping from output to
        # (initially empty) lists to hold predictions
        y_pred = {o: [] for o in outputs}

        # Loop through batches and evaluate
        for batch, x_batch in enumerate(x):
            # Start batch
            callback_list.on_batch_begin(batch)

            # Get predictions from each output on this batch
            y_pred_batch = self._predict_batch(batch, x_batch, outputs,
                                               callback_list)

            # Insert copies into dictionary
            for o, y in  zip(outputs, y_pred_batch):
                y_pred[o].append(np.copy(y))

            # End batch
            callback_list.on_batch_end(batch, {})

        # End testing
        callback_list.on_test_end({})

        # Concatenate predictions into single numpy array and trim padding
        for o in outputs:
            y_pred[o] = np.concatenate(y_pred[o])[:x_size,:]

        # Return predictions and metrics
        return y_pred, callback_list.get_data()

    def evaluate_batch(self, x: dict, y: dict,
                       metrics="sparse_categorical_accuracy",
                       callbacks=[]):
        # Build metrics
        metrics = get_object_mapping(metrics, y.keys(), Metric, 
                                     "Metric", default_metrics)

        # Create callback list and begin testing
        callback_list = CallbackList(self.base_callbacks + callbacks,
                                     compiled_network=self,
                                     num_batches=1)
        callback_list.on_test_begin()

        # Evaluate batch and return metrics
        self._evaluate_batch(0, x, y, metrics, callback_list)

        # End testing
        callback_list.on_test_end(metrics)

        return metrics, callback_list.get_data()

    def _predict_batch(self, batch: int, x: dict, outputs: Sequence,
                       callback_list: CallbackList):
        """ Generate predictions from a single batch of inputs
        Args:
        batch --    index of current batch
        x --        dict mapping input Population or InputLayer to
                    array containing one batch of inputs
        outputs --  sequence of populations to read predictions from
        """
        # Reset time to 0
        # **YUCK** I don't REALLY like this
        self.genn_model.timestep = 0

        # Set x as input
        self.set_input(x)

        # Simulate timesteps
        for t in range(self.evaluate_timesteps):
            self.step_time(callback_list)

        # Return predictions from model
        return self.get_readout(outputs)

    def _evaluate_batch(self, batch: int, x: dict, y: dict, metrics,
                        callback_list: CallbackList):
        """ Evaluate a single batch of inputs against labels
        Args:
        batch --    index of current batch
        x --        dict mapping input Population or InputLayer to
                    array containing one batch of inputs
        y --        dict mapping output Population or Layer to
                    array containing one batch of labels
        """
        # Start batch
        callback_list.on_batch_begin(batch)

        # Get predictions from model
        y_pred = self._predict_batch(batch, x, list(y.keys()), callback_list)

        # Update metrics
        for (o, y_true), out_y_pred in zip(y.items(), y_pred):
            metrics[o].update(y_true, out_y_pred[:len(y_true)],
                              self.communicator)

        # End batch
        callback_list.on_batch_end(batch, metrics)

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
        self.evaluate_timesteps = k

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
                    con_delay[c()] = depth_difference * self.evaluate_timesteps

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
                                       self.communicator, self.evaluate_timesteps,
                                       compile_state.pop_pipeline_depth)
