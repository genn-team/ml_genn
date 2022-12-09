import numpy as np

from .compiled_network import CompiledNetwork
from ..callbacks import BatchProgressBar
from ..connectivity.sparse_base import SparseBase
from ..metrics import Metric
from ..serialisers import Serialiser
from ..utils.callback_list import CallbackList
from ..utils.data import MetricsType

from ..utils.data import batch_dataset, get_dataset_size, permute_dataset
from ..utils.module import get_object, get_object_mapping
from ..utils.network import get_underlying_pop

from ..metrics import default_metrics
from ..serialisers import default_serialisers

class CompiledTrainingNetwork(CompiledNetwork):
    def __init__(self, genn_model, neuron_populations,
                 connection_populations, softmax, losses,
                 optimiser, example_timesteps: int, base_callbacks: list,
                 optimiser_custom_updates: list,
                 checkpoint_connection_vars: list,
                 checkpoint_population_vars: list,
                 reset_time_between_batches: bool = True):
        super(CompiledTrainingNetwork, self).__init__(
            genn_model, neuron_populations, connection_populations,
            softmax, example_timesteps)

        self.losses = losses
        self.optimiser = optimiser
        self.example_timesteps = example_timesteps
        self.base_callbacks = base_callbacks
        self.optimiser_custom_updates = optimiser_custom_updates
        self.checkpoint_connection_vars = checkpoint_connection_vars
        self.checkpoint_population_vars = checkpoint_population_vars
        self.reset_time_between_batches = reset_time_between_batches
        
        # Build set of synapse groups with checkpoint variables
        self.checkpoint_synapse_groups = set(
            connection_populations[c] 
            for c, _ in self.checkpoint_connection_vars)

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

        # Create callback list and begin testing
        batch_size = self.genn_model.batch_size
        num_batches = (x_size + batch_size - 1) // batch_size;
        callback_list = CallbackList(self.base_callbacks + callbacks,
                                     compiled_network=self,
                                     num_batches=num_batches, 
                                     num_epochs=num_epochs)
        callback_list.on_train_begin()

        # Loop through epochs
        step_i = 0
        for e in range(num_epochs):
            # If we should shuffle
            if shuffle:
                # Generate random permutation
                indices = np.random.permutation(x_size)

                # Permute x and y
                x = permute_dataset(x, indices)
                y = permute_dataset(y, indices)

            # Batch x and y
            xy_batched = zip(batch_dataset(x, batch_size, x_size),
                             batch_dataset(y, batch_size, y_size))

            # Reset metrics at start of each epoch
            for m in metrics.values():
                m.reset()

            callback_list.on_epoch_begin(e)

            # Loop through batches and train
            for batch_i, (x_batch, y_batch) in enumerate(xy_batched):
                self._train_batch(batch_i, step_i, x_batch, y_batch,
                                  metrics, callback_list)
                step_i += 1

            callback_list.on_epoch_end(e, metrics)

        # End testing
        callback_list.on_train_end(metrics)

        # Return metrics
        return metrics, callback_list.get_data()

    def save_connectivity(self, keys=(), serialiser="numpy"):
        # Create serialiser
        serialiser = get_object(serialiser, Serialiser, "Serialiser",
                                default_serialisers)
        
        # Loop through connections and their corresponding synapse groups
        for c, genn_pop in self.connection_populations.items():
            # If synapse group has ragged connectivity, download  
            # connectivity and save pre and postsynaptic indices
            if genn_pop.is_ragged:
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
            # If synapse group has ragged connectivity, download  
            # connectivity so variables can be accessed correctly
            if genn_pop.is_ragged:
                genn_pop.pull_connectivity_from_device()
                
        # Loop through connection variables to checkpoint
        for c, v in self.checkpoint_connection_vars:
            genn_pop = self.connection_populations[c]
            genn_pop.pull_var_from_device(v)
            serialiser.serialise(keys + (c, v), genn_pop.get_var_values(v))

        # Loop through population variables to checkpoint
        for p, v in self.checkpoint_population_vars:
            genn_pop = self.neuron_populations[p]
            genn_pop.pull_var_from_device(v)
            serialiser.serialise(keys + (p, v), genn_pop.vars[v].view)

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
            metrics[o].update(y_true, out_y_pred[:len(y_true)])

        # Loop through optimiser custom updates and set step
        for c in self.optimiser_custom_updates:
            self.optimiser.set_step(c, step)

        # End batch
        callback_list.on_batch_end(batch, metrics)
