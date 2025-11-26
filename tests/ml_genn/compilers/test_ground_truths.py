import numpy as np
import pytest

from ml_genn.compilers.ground_truths import (ExampleLabel, ExampleValue,
                                             TimestepValue)
from unittest.mock import MagicMock,Mock, PropertyMock

from ml_genn.utils.data import batch_dataset
from pytest import raises

def mock_genn_pop(batch_size, var_shape=None,
                  egp_size=None):
    assert var_shape or egp_size
    
    if var_shape is not None:
        var_size = np.prod(var_shape)
        view_shape = ((batch_size, var_size) 
                      if batch_size > 1 else var_size)
    else:
        view_shape = batch_size * egp_size
    
    # Mock variable object for y_true so it's 
    # view is a correctly-shaped numpy array
    y_true = Mock()
    type(y_true).view = PropertyMock(return_value=np.empty(view_shape))

    # Mock genn_pop
    genn_pop = MagicMock()

    # Configure either vars or egps so they return dictionary
    d = {"YTrue": y_true}
    if var_shape is not None:
        genn_pop.vars.__getitem__.side_effect = d.__getitem__
    if egp_size is not None:
        genn_pop.extra_global_params.__getitem__.side_effect = d.__getitem__
    return genn_pop

@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("shape", [(10,), (2, 5)])
def test_example_label(batch_size, shape):
    # Create some labels and batch
    y_true = {"output": np.ones(40, dtype=int)}
    y_true_batch = batch_dataset(y_true, batch_size, 40)

    # Mock a GeNN model
    genn_pop = mock_genn_pop(batch_size, var_shape=(1,))

    # Create ground truth
    ground_truth = ExampleLabel()

    # Loop through batches and push
    for y in y_true_batch:
        ground_truth.push_to_device(genn_pop, y["output"], 
                                    shape, batch_size, 30)

@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("shape", [(10,), (2, 5)])
def test_example_value(batch_size, shape):
    # Create some labels and batch
    y_true = {"output": np.ones((40,) + shape, dtype=int)}
    y_true_batch = batch_dataset(y_true, batch_size, 40)

    # Mock a GeNN model
    genn_pop = mock_genn_pop(batch_size, var_shape=shape)

    # Create ground truth
    ground_truth = ExampleValue()

    # Loop through batches and push
    for y in y_true_batch:
        ground_truth.push_to_device(genn_pop, y["output"], 
                                    shape, batch_size, 30)

@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("shape", [(10,), (2, 5)])
def test_timestep_value(batch_size, shape):
    # Create some labels and batch
    y_true = {"output": np.ones((40, 30) + shape, dtype=int)}
    y_true_batch = batch_dataset(y_true, batch_size, 40)

    # Mock a GeNN model
    genn_pop = mock_genn_pop(batch_size, egp_size=30 * np.prod(shape))

    # Create ground truth
    ground_truth = TimestepValue()

    # Loop through batches and push
    for y in y_true_batch:
        ground_truth.push_to_device(genn_pop, y["output"], 
                                    shape, batch_size, 30)