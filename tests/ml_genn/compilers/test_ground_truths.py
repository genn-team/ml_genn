import numpy as np
import pytest

from ml_genn.compilers.ground_truths import (ExampleLabel, ExampleValue,
                                             TimestepValue)
from unittest.mock import MagicMock

from ml_genn.utils.data import batch_dataset
from pytest import raises

@pytest.mark.parametrize("batch_size", [1, 32])
def test_example_label(batch_size):
    # Create some labels and batch
    y_true = {"output": np.ones(40, dtype=int)}
    y_true_batch = batch_dataset(y_true, batch_size, 40)

    # Mock a GeNN model
    genn_pop = MagicMock()

    # Create ground truth
    ground_truth = ExampleLabel()

    # Loop through batches and push
    for y in y_true_batch:
        ground_truth.push_to_device(genn_pop, y["output"], 
                                    (10,), batch_size, 30)

@pytest.mark.parametrize("batch_size", [1, 32])
def test_example_value(batch_size):
    # Create some labels and batch
    y_true = {"output": np.ones((40, 10), dtype=int)}
    y_true_batch = batch_dataset(y_true, batch_size, 40)

    # Mock a GeNN model
    genn_pop = MagicMock()

    # Create ground truth
    ground_truth = ExampleValue()

    # Loop through batches and push
    for y in y_true_batch:
        ground_truth.push_to_device(genn_pop, y["output"], 
                                    (10,), batch_size, 30)

@pytest.mark.parametrize("batch_size", [1, 32])
def test_timestep_value(batch_size):
    # Create some labels and batch
    y_true = {"output": np.ones((40, 30, 10), dtype=int)}
    y_true_batch = batch_dataset(y_true, batch_size, 40)

    # Mock a GeNN model
    genn_pop = MagicMock()

    # Create ground truth
    ground_truth = TimestepValue()

    # Loop through batches and push
    for y in y_true_batch:
        ground_truth.push_to_device(genn_pop, y["output"], 
                                    (10,), batch_size, 30)