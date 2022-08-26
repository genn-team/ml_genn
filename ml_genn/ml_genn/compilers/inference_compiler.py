from typing import Iterator, Sequence
from pygenn.genn_wrapper.Models import VarAccessMode_READ_WRITE
from .compiler import Compiler
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType
from ..utils.model import CustomUpdateModel

from functools import partial
from pygenn.genn_model import create_var_ref, create_psm_var_ref
from .compiler import build_model
from ..utils.data import batch_dataset, get_metrics, get_dataset_size


def _build_reset_model(model, custom_updates, var_ref_creator):
    # If model has any state variables
    reset_vars = []
    if "var_name_types" in model.model:
        # Loop through them
        for v in model.model["var_name_types"]:
            # If variable either has default (read-write)
            # access or this is explicitly set
            # **TODO** mechanism to exclude variables from reset
            if len(v) < 3 or (v[2] & VarAccessMode_READ_WRITE) != 0:
                reset_vars.append((v[0], v[1], model.var_vals[v[0]]))

    # If there's nothing to reset, return
    if len(reset_vars) == 0:
        return

    # Create empty model
    model = CustomUpdateModel(model={"param_name_types": [],
                                     "var_refs": [],
                                     "update_code": ""},
                              param_vals={}, var_vals={}, var_refs={})

    # Loop through reset vars
    for name, type, value in reset_vars:
        # Add variable reference and reset value parameter
        model.model["var_refs"].append((name, type))
        model.model["param_name_types"].append((name + "Reset", type))

        # Add parameter value and function to create variable reference
        # **NOTE** we use partial rather than a lambda so name is captured
        model.param_vals[name + "Reset"] = value
        model.var_refs[name] = partial(var_ref_creator, name=name)

        # Add code to set var
        model.model["update_code"] += f"$({name}) = $({name}Reset);\n"

    # Build custom update model customised for parameters and values
    custom_model = build_model(model)

    # Add to list of custom updates to be applied in "Reset" group
    custom_updates["Reset"].append(custom_model + (model.var_refs,))


class CompiledInferenceNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, evaluate_timesteps: int,
                 reset_time_between_batches: bool = True):
        super(CompiledInferenceNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations,
            evaluate_timesteps)
        
        self.evaluate_timesteps = evaluate_timesteps
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
        metrics = get_metrics(metrics, y.keys())

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
        callback_list = CallbackList(callbacks, compiled_network=self,
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
        metrics = get_metrics(metrics, outputs)

        # Create callback list and begin testing
        callback_list = CallbackList(callbacks, compiled_network=self,
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
        metrics = get_metrics(metrics, y.keys())

        # Create callback list and begin testing
        callback_list = CallbackList(callbacks)
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

        self.custom_update("Reset")
        
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
        y_pred = self.get_output(list(y.keys()))

        # Update metrics
        for (o, y_true), out_y_pred in zip(y.items(), y_pred):
            metrics[o].update(y_true, out_y_pred[:len(y_true)])

        # End batch
        callback_list.on_batch_end(batch, metrics)


class InferenceCompiler(Compiler):
    def __init__(self, evaluate_timesteps: int, dt: float = 1.0,
                 batch_size: int = 1, rng_seed: int = 0,
                 kernel_profiling: bool = False,
                 prefer_in_memory_connect=True, 
                 reset_time_between_batches=True,
                 **genn_kwargs):
        super(InferenceCompiler, self).__init__(dt, batch_size, rng_seed,
                                                kernel_profiling,
                                                prefer_in_memory_connect,
                                                **genn_kwargs)
        self.evaluate_timesteps = evaluate_timesteps
        self.reset_time_between_batches = reset_time_between_batches

    def build_neuron_model(self, pop, model, custom_updates,
                           pre_compile_output):
        _build_reset_model(
            model, custom_updates,
            lambda n_pops, _, name: create_var_ref(n_pops[pop], name))

        # Build neuron model
        return super(InferenceCompiler, self).build_neuron_model(
            pop, model, custom_updates, pre_compile_output)

    def build_synapse_model(self, conn, model, custom_updates,
                            pre_compile_output):
        _build_reset_model(
            model, custom_updates,
            lambda _, c_pops, name: create_psm_var_ref(c_pops[conn], name))

        return super(InferenceCompiler, self).build_synapse_model(
            conn, model, custom_updates, pre_compile_output)

    def create_compiled_network(self, genn_model, neuron_populations,
                                connection_populations, pre_compile_output):
        return CompiledInferenceNetwork(genn_model, neuron_populations,
                                        connection_populations,
                                        self.evaluate_timesteps,
                                        self.reset_time_between_batches)
