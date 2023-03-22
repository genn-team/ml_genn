from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler, ZeroInSyn
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar, CustomUpdateOnBatchBegin
from ..metrics import Metric
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType
from ..utils.model import CustomUpdateModel

from pygenn.genn_model import create_var_ref, create_psm_var_ref
from .compiler import create_reset_custom_update
from ..utils.data import batch_dataset, get_dataset_size
from ..utils.module import get_object_mapping

from ..metrics import default_metrics


class CompiledInferenceNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, evaluate_timesteps: int, 
                 base_callbacks: list,
                 reset_time_between_batches: bool = True):
        super(CompiledInferenceNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations,
            evaluate_timesteps)
        
        self.evaluate_timesteps = evaluate_timesteps
        self.base_callbacks = base_callbacks
        self.reset_time_between_batches = reset_time_between_batches

    def evaluate(self, x: dict, y: dict,
                 metrics: MetricsType = "sparse_categorical_accuracy",
                 callbacks=[BatchProgressBar()]):
        """ Evaluate an input in numpy format against labels
        accuracy --  dictionary containing accuracy of predictions
                     made by each output Population or Layer
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
        callback_list = CallbackList(self.base_callbacks + callbacks,
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

    def evaluate_batch_iter(
            self, inputs, outputs, data: Iterator, num_batches: int = None,
            metrics: MetricsType = "sparse_categorical_accuracy",
            callbacks=[BatchProgressBar()]):
        """ Evaluate an input in iterator format against labels
        accuracy --  dictionary containing accuracy of predictions
                     made by each output Population or Layer
        """
        # Convert inputs and outputs to tuples
        inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
        outputs = outputs if isinstance(outputs, Sequence) else (outputs,)

        # Build metrics
        metrics = get_object_mapping(metrics, outputs, Metric, 
                                     "Metric", default_metrics)

        # Create callback list and begin testing
        callback_list = CallbackList(self.base_callbacks + callbacks,
                                     compiled_network=self,
                                     num_batches=num_batches)
        callback_list.on_test_begin()

        # Loop through data
        batch_i = 0
        while True:
            # Attempt to get next batch of data,
            # break if none remains
            try:
                batch_x, batch_y = next(data)
            except StopIteration:
                break

            # Set x as input
            # **YUCK** this isn't quite right as batch_x
            # could also have outer dimension
            if len(inputs) == 1:
                x = {inputs[0]: batch_x}
            else:
                x = {p: x for p, x in zip(inputs, batch_x)}

            # Add each y to correct queue(s)
            # **YUCK** this isn't quite right as batch_y
            # could also have outer dimension
            if len(outputs) == 1:
                y = {outputs[0]: batch_y}
            else:
                y = {p: y for p, x in zip(outputs, batch_y)}

            # Evaluate batch
            self._evaluate_batch(batch_i, x, y, metrics, callback_list)
            batch_i += 1

        # End testing
        callback_list.on_test_end(metrics)

        # Return metrics
        return metrics, callback_list.get_data()

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

    def _evaluate_batch(self, batch: int, x: dict, y: dict, metrics,
                        callback_list: CallbackList):
        """ Evaluate a single batch of inputs against labels
        Args:
        batch --    index of current batch
        x --        dict mapping input Population or InputLayer to
                    array containing one batch of inputs
        y --        dict mapping output Population or Layer to
                    array containing one batch of labels

        Returns:
        correct --  dictionary containing number of correct predictions
                    made by each output Population or Layer
        """
        # Start batch
        callback_list.on_batch_begin(batch)

        # Reset time to 0 if desired
        if self.reset_time_between_batches:
            self.genn_model.timestep = 0
            self.genn_model.t = 0.0

        # Apply inputs to model
        self.set_input(x)

        # Simulate timesteps
        for t in range(self.evaluate_timesteps):
            self.step_time(callback_list)

        # Get predictions from model
        y_pred = self.get_readout(list(y.keys()))

        # Update metrics
        for (o, y_true), out_y_pred in zip(y.items(), y_pred):
            metrics[o].update(y_true, out_y_pred[:len(y_true)])

        # End batch
        callback_list.on_batch_end(batch, metrics)


class CompileState:
    def __init__(self):
        self._neuron_reset_vars = {}
        self._psm_reset_vars = {}
    
    def add_neuron_reset_vars(self, model, pop, reset_model_vars):
        if reset_model_vars:
            reset_vars = model.reset_vars
        elif pop.neuron.readout is not None:
            reset_vars = pop.neuron.readout.reset_vars
        else:
            reset_vars = []

        if len(reset_vars) > 0:
            self._neuron_reset_vars[pop] = reset_vars

    def add_psm_reset_vars(self, model, conn):
        reset_vars = model.reset_vars
        if len(reset_vars) > 0:
            self._psm_reset_vars[conn] = reset_vars
    
    def create_reset_custom_updates(self, compiler, genn_model,
                                    neuron_pops, conn_pops):
        # Loop through neuron variables to reset
        for i, (pop, reset_vars) in enumerate(self._neuron_reset_vars.items()):
            # Create reset model
            model = create_reset_custom_update(
                reset_vars,
                lambda name: create_var_ref(neuron_pops[pop], name))

            # Add custom update
            compiler.add_custom_update(genn_model, model, 
                                       "Reset", f"CUResetNeuron{i}")

        # Loop through psm variables to reset
        for i, (conn, reset_vars) in enumerate(self._psm_reset_vars.items()):
            # Create reset model
            model = create_reset_custom_update(
                reset_vars,
                lambda name: create_psm_var_ref(conn_pops[conn], name))

            # Add custom update
            compiler.add_custom_update(genn_model, model, 
                                       "Reset", f"CUResetPSM{i}")


class InferenceCompiler(Compiler):
    def __init__(self, evaluate_timesteps: int, dt: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 prefer_in_memory_connect=True, 
                 reset_time_between_batches=True,
                 reset_vars_between_batches=True,
                 reset_in_syn_between_batches=False,
                 **genn_kwargs):
        super(InferenceCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling,
                                                prefer_in_memory_connect,
                                                **genn_kwargs)
        self.evaluate_timesteps = evaluate_timesteps
        self.reset_time_between_batches = reset_time_between_batches
        self.reset_vars_between_batches = reset_vars_between_batches
        self.reset_in_syn_between_batches = reset_in_syn_between_batches

    def pre_compile(self, network, **kwargs):
        return CompileState()

    def build_neuron_model(self, pop, model, compile_state):
        # If population has a readout i.e. it's an output
        # Add readout logic to model
        if pop.neuron.readout is not None:
            model = pop.neuron.readout.add_readout_logic(
                model, example_timesteps=self.evaluate_timesteps,
                dt=self.dt)

        # Add any neuron reset variables to compile state
        compile_state.add_neuron_reset_vars(model, pop,
                                            self.reset_vars_between_batches)

        # Build neuron model
        return super(InferenceCompiler, self).build_neuron_model(
            pop, model, compile_state)

    def build_synapse_model(self, conn, model, compile_state):
        # Add any PSM reset variables to compile state
        if self.reset_vars_between_batches:
            compile_state.add_psm_reset_vars(model, conn)

        return super(InferenceCompiler, self).build_synapse_model(
            conn, model, compile_state)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, compile_state):
        # Create custom updates to implement variable reset
        compile_state.create_reset_custom_updates(self, genn_model,
                                                  neuron_populations,
                                                  connection_populations)
        
        base_callbacks = [CustomUpdateOnBatchBegin("Reset")]
        
        # Add callbacks to zero insyn on all connections
        # **NOTE** it would be great to be able to do this on device
        for genn_syn_pop in connection_populations.values():
            base_callbacks.append(ZeroInSyn(genn_syn_pop,
                                            self.evaluate_timesteps))
    

        return CompiledInferenceNetwork(genn_model, neuron_populations,
                                        connection_populations,
                                        self.evaluate_timesteps,
                                        base_callbacks,
                                        self.reset_time_between_batches)
