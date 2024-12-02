import numpy as np

from typing import List, Optional, Tuple
from pygenn import SynapseMatrixConnectivity
from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..connectivity.sparse_base import SparseBase
from ..metrics import Metric, MetricsType
from ..serialisers import Serialiser
from ..utils.callback_list import CallbackList

from ..utils.data import (batch_dataset, get_dataset_size,
                          permute_dataset, split_dataset)
from ..utils.module import get_object, get_object_mapping
from ..utils.network import get_underlying_pop

from ..metrics import default_metrics
from ..serialisers import default_serialisers

class CompiledTrainingNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, communicator,
                 losses, example_timesteps: int,
                 base_train_callbacks: list, base_validate_callbacks: list,
                 optimisers: List[Tuple],
                 checkpoint_connection_vars: list,
                 checkpoint_population_vars: list,
                 reset_time_between_batches: bool = True):
        super(CompiledTrainingNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations,
            communicator, example_timesteps)

        self.losses = losses
        self.example_timesteps = example_timesteps
        self.base_train_callbacks = base_train_callbacks
        self.base_validate_callbacks = base_validate_callbacks
        self.optimisers = optimisers
        self.checkpoint_connection_vars = checkpoint_connection_vars
        self.checkpoint_population_vars = checkpoint_population_vars
        self.reset_time_between_batches = reset_time_between_batches

        # Build set of synapse groups with checkpoint variables
        self.checkpoint_synapse_groups = set(
            connection_populations[c] 
            for c, _ in self.checkpoint_connection_vars)

    def train(self, x: dict, y: dict, num_epochs: int, 
              start_epoch: int = 0, shuffle: bool = True,
              metrics: MetricsType = "sparse_categorical_accuracy",
              callbacks=[BatchProgressBar()],
              validation_callbacks=[BatchProgressBar()],
              validation_split: float = 0.0,
              validation_x: Optional[dict] = None, 
              validation_y: Optional[dict] = None):
        """ Train model on an input in numpy format against labels

        Args:
            x:                      Dictionary of training inputs
            y:                      Dictionary of training labels 
                                    to compare predictions against
            num_epochs:             Number of epochs to train for
            start_epoch:            Epoch to stasrt training from
            shuffle:                Should training data be shuffled
                                    between epochs?
            metrics:                Metrics to calculate.
            callbacks:              List of callbacks to run during training.
            validation_callbacks:   List of callbacks to run during validation
            validation_split:       Float between 0 and 1 specifying the
                                    fraction of the training data to use
                                    for validation.
            validation_x:           Dictionary of validation inputs (cannot be
                                    used at same time as ``validation_split``)
            validation_y:           Dictionary of validation labels (cannot be
                                    used at same time as ``validation_split``)
        """        
        # If validation split is specified
        if validation_split != 0.0:
            # Check validation data isn't also provided
            if validation_x is not None or validation_y is not None:
                raise RuntimeError("validation data and validation split "
                                   "cannot both be provided")

            # Split dataset into training and validation
            x, validation_x = split_dataset(x, validation_split)
            y, validation_y = split_dataset(y, validation_split)

        # Get the size of x and y
        x_train_size = get_dataset_size(x)
        y_train_size = get_dataset_size(y)

        # Check training sizes match
        if x_train_size is None:
            raise RuntimeError("Each input population must be "
                               "provided with same number of training inputs")
        if y_train_size is None:
            raise RuntimeError("Each output population must be "
                               "provided with same number of training labels")
        if x_train_size != y_train_size:
            raise RuntimeError("Number of training inputs "
                               "and labels must match")

        # If seperate validation data is provided
        if validation_x is not None and validation_y is not None:
            # Get the size of validation_x and validation_y
            x_validate_size = get_dataset_size(validation_x)
            y_validate_size = get_dataset_size(validation_y)

            # Check validation sizes match
            if x_validate_size is None:
                raise RuntimeError("Each input population must be provided "
                                   "with same number of validation inputs")
            if y_validate_size is None:
                raise RuntimeError("Each output population must be provided "
                                   "with same number of validation labels")
            if x_validate_size != y_validate_size:
                raise RuntimeError("Number of validation inputs "
                                   "and labels must match")
        # Otherwise, no validation is required
        else:
            x_validate_size = 0
            y_validate_size = 0

        # Build metrics for training
        train_metrics = get_object_mapping(metrics, y.keys(), Metric, 
                                           "Metric", default_metrics)

        # Create callback list
        batch_size = self.genn_model.batch_size
        num_train_batches = (x_train_size + batch_size - 1) // batch_size
        train_callback_list = CallbackList(
            self.base_train_callbacks + callbacks,
            compiled_network=self,
            num_batches=num_train_batches, 
            num_epochs=num_epochs)
        train_callback_list.on_train_begin()
        
        # If there's any validation data
        if x_validate_size > 0:
            # Build metrics for validation
            validate_metrics = get_object_mapping(metrics, y.keys(), Metric, 
                                                  "Metric", default_metrics)

            # Create seperate callback list and begin testing
            num_validate_batches = (x_validate_size + batch_size - 1) // batch_size
            validate_callback_list = CallbackList(
                self.base_validate_callbacks + validation_callbacks,
                compiled_network=self,
                num_batches=num_validate_batches, 
                num_epochs=num_epochs)
            validate_callback_list.on_test_begin()
            
            # Batch validation data
            # **NOTE** need to turn zip into list 
            # as, otherwise, it can only be iterated once
            xy_validate_batched = list(zip(
                batch_dataset(validation_x, batch_size, x_validate_size),
                batch_dataset(validation_y, batch_size, y_validate_size)))

        # Loop through epochs
        step_i = num_train_batches * start_epoch
        for e in range(start_epoch, start_epoch + num_epochs):
            # If we should shuffle
            if shuffle:
                # Generate random permutation
                indices = np.random.permutation(x_train_size)

                # Permute x and y
                x = permute_dataset(x, indices)
                y = permute_dataset(y, indices)

            # Batch x and y
            # **NOTE** can use zip directly as 
            # it only needs to be iterated once
            xy_train_batched = zip(batch_dataset(x, batch_size, x_train_size),
                                   batch_dataset(y, batch_size, y_train_size))

            # Reset training metrics at start of each epoch
            for m in train_metrics.values():
                m.reset()

            train_callback_list.on_epoch_begin(e)

            # Loop through batches and train
            for batch_i, (x_batch, y_batch) in enumerate(xy_train_batched):
                self._train_batch(batch_i, step_i, x_batch, y_batch,
                                  train_metrics, train_callback_list)
                step_i += 1

            train_callback_list.on_epoch_end(e, train_metrics)

            # If there's any validation data
            if x_validate_size > 0:
                # Reset validation metrics at start of each epoch
                for m in validate_metrics.values():
                    m.reset()

                validate_callback_list.on_epoch_begin(e)

                # Loop through batches and validate
                for batch_i, (x_batch, y_batch) in enumerate(xy_validate_batched):
                    self._validate_batch(batch_i, x_batch, y_batch,
                                         validate_metrics, validate_callback_list)

                validate_callback_list.on_epoch_end(e, validate_metrics)

        # End training
        train_callback_list.on_train_end(train_metrics)

        # If there's any validation data
        if x_validate_size > 0:
            validate_callback_list.on_test_end(validate_metrics)
            
            return (train_metrics, validate_metrics,
                    train_callback_list.get_data(),
                    validate_callback_list.get_data())
        # Otherwise, just return training metrics and callback data
        else:
            return train_metrics, train_callback_list.get_data()

    def save_connectivity(self, keys=(), serialiser="numpy"):
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)
        
        # Loop through connections and their corresponding synapse groups
        for c, genn_pop in self.connection_populations.items():
            # If synapse group has sparse connectivity, download  
            # connectivity and save pre and postsynaptic indices
            if genn_pop.matrix_type & SynapseMatrixConnectivity.SPARSE:
                genn_pop.pull_connectivity_from_device()
                serialiser.serialise(keys + (c, "pre_ind"),
                                     genn_pop.get_sparse_pre_inds())
                serialiser.serialise(keys + (c, "post_ind"),
                                     genn_pop.get_sparse_post_inds())

    def save(self, keys=(), serialiser="numpy"):
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)
        
        # Loop through synapse groups with variables to be checkpointed
        for genn_pop in self.checkpoint_synapse_groups:
            # If synapse group has sparse connectivity, download  
            # connectivity so variables can be accessed correctly
            if genn_pop.matrix_type & SynapseMatrixConnectivity.SPARSE:
                genn_pop.pull_connectivity_from_device()

        # Loop through connection variables to checkpoint
        for c, v in self.checkpoint_connection_vars:
            genn_var = self.connection_populations[c].vars[v]
            genn_var.pull_from_device()
            serialiser.serialise(keys + (c, v), genn_var.values)

        # Loop through population variables to checkpoint
        for p, v in self.checkpoint_population_vars:
            genn_var = self.neuron_populations[p].vars[v]
            genn_var.pull_from_device()
            serialiser.serialise(keys + (p, v), genn_var.values)
    
    def _validate_batch(self, batch: int, x: dict, y: dict, metrics,
                        callback_list: CallbackList):
        # Start batch
        callback_list.on_batch_begin(batch)

        # Reset time to 0 if desired
        if self.reset_time_between_batches:
            self.genn_model.timestep = 0

        # Apply inputs to model
        self.set_input(x)

        # Simulate timesteps
        for t in range(self.example_timesteps):
            self.step_time(callback_list)

        # Get predictions from model
        y_pred = self.get_readout(list(y.keys()))

        # Update metrics
        for (o, y_true), out_y_pred in zip(y.items(), y_pred):
            metrics[o].update(y_true, out_y_pred[:len(y_true)],
                              self.communicator)

        # End batch
        callback_list.on_batch_end(batch, metrics)

    def _train_batch(self, batch: int, step: int, x: dict, y: dict,
                     metrics, callback_list: CallbackList):
        # Start batch
        callback_list.on_batch_begin(batch)

        # Reset time to 0 if desired
        if self.reset_time_between_batches:
            self.genn_model.timestep = 0

        # Apply inputs to model
        self.set_input(x)

        # Loop through outputs
        for pop, y_true in y.items():
            # Update loss function with target labels
            underlying_pop = get_underlying_pop(pop)
            self.losses[underlying_pop].set_target(
                self.neuron_populations[underlying_pop],
                y_true, underlying_pop.shape, self.genn_model.batch_size,
                self.example_timesteps)
 
        # Simulate timesteps
        for t in range(self.example_timesteps):
            self.step_time(callback_list)

        # Get predictions from model
        y_pred = self.get_readout(list(y.keys()))

        # Update metrics
        for (o, y_true), out_y_pred in zip(y.items(), y_pred):
            metrics[o].update(y_true, out_y_pred[:len(y_true)],
                              self.communicator)

        # Loop through optimisers
        for o, custom_updates in self.optimisers:
            # Set step on all custom updates
            for c in custom_updates:
                o.set_step(c, step)

        # End batch
        callback_list.on_batch_end(batch, metrics)
