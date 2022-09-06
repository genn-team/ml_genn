import random

from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..metrics import Metric
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType

from ..utils.data import batch_dataset, get_dataset_size
from ..utils.module import get_object_mapping
from ..utils.network import get_underlying_pop

from ..metrics import default_metrics

class CompiledTrainingNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, losses, 
                 optimiser, example_timesteps: int, 
                 reset_time_between_batches: bool = True):
        super(CompiledTrainingNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations,
            example_timesteps)
        
        self.losses = losses
        self.optimiser = optimiser
        self.example_timesteps = example_timesteps
        self.reset_time_between_batches = reset_time_between_batches
        
        # **YUCK** find optimiser custom updates
        self._optimizer_custom_updates = [
            c for c in genn_model.custom_updates.values()
            if c.pop.get_update_group_name() == "GradientLearn"]

    def train(self, x: dict, y: dict, num_epochs: int, shuffle: bool = True,
              metrics: MetricsType = "sparse_categorical_accuracy",
              callbacks=[BatchProgressBar()]):
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

        # Zip together x and y
        xy = list(zip(x, y))
        
        # Create callback list and begin testing
        callback_list = CallbackList(callbacks, compiled_network=self,
                                     num_batches=len(xy), num_epochs=num_epochs)
        callback_list.on_train_begin()

        # Loop through epochs
        step_i = 0
        for e in range(num_epochs):
            # If we should shuffle, do so
            if shuffle:
                random.shuffle(xy)

            callback_list.on_epoch_begin(e)

            # Loop through batches and train
            for batch_i, (x_batch, y_batch) in enumerate(xy):
                self._train_batch(batch_i, step_i, x_batch, y_batch,
                                  metrics, callback_list)
                step_i += 1

            callback_list.on_epoch_end(e, metrics)

        # End testing
        callback_list.on_train_end(metrics)

        # Return metrics
        return metrics, callback_list.get_data()
    """
    def train_batch_iter(
            self, inputs, outputs, data: Iterator, num_batches: int = None,
            metrics: MetricsType = "sparse_categorical_accuracy",
            callbacks=[BatchProgressBar()]):
        # Convert inputs and outputs to tuples
        inputs = inputs if isinstance(inputs, Sequence) else (inputs,)
        outputs = outputs if isinstance(outputs, Sequence) else (outputs,)

        # Build metrics
        metrics = get_object_mapping(metrics, outputs, Metric, 
                                     "Metric", default_metrics)

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

    def train_batch(self, x: dict, y: dict,
                    metrics="sparse_categorical_accuracy",
                    callbacks=[]):
        # Build metrics
        metrics = get_object_mapping(metrics, y.keys(), Metric, 
                                     "Metric", default_metrics)

        # Create callback list and begin testing
        callback_list = CallbackList(callbacks)
        callback_list.on_test_begin()

        # Evaluate batch and return metrics
        self._evaluate_batch(0, x, y, metrics, callback_list)

        # End testing
        callback_list.on_test_end(metrics)

        return metrics, callback_list.get_data()
    """
    def _train_batch(self, batch: int, step: int,
                     x: dict, y: dict, metrics,
                     callback_list: CallbackList):
        # Start batch
        callback_list.on_batch_begin(batch)

        # Reset time to 0 if desired
        if self.reset_time_between_batches:
            self.genn_model.timestep = 0
            self.genn_model.t = 0.0

        # Apply inputs to model
        self.set_input(x)

        # Loop through loss functions
        for pop, loss in self.losses.items():
            # Find corresponding GeNN population
            pop = get_underlying_pop(pop)

            # Update loss function with target labels
            loss.set_target(self.neuron_populations[pop], y[pop],
                            pop.shape, self.genn_model.batch_size,
                            self.example_timesteps)
 
        # Simulate timesteps
        for t in range(self.example_timesteps):
            self.step_time(callback_list)

        # Get predictions from model
        y_pred = self.get_output(list(y.keys()))

        # Update metrics
        for (o, y_true), out_y_pred in zip(y.items(), y_pred):
            metrics[o].update(y_true, out_y_pred[:len(y_true)])

        # Loop through optimiser custom updates and set step
        # **TODO** mechanism for setting learning rate
        for c in self._optimizer_custom_updates:
            self.optimiser.set_step(c, step)

        # Now batch is complete, apply gradients
        # **YUCK** this needs to be more generic - probably use callbacks
        self.genn_model.custom_update("GradientLearn")

        # End batch
        callback_list.on_batch_end(batch, metrics)