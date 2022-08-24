import numpy as np

from typing import Optional, Union
from ..metrics import Metric

from copy import deepcopy

from ..metrics import default_metrics

MetricType = Union[Metric, str]
MetricsType = Union[dict, MetricType]


def get_metric(metric: MetricType) -> Metric:
    if isinstance(metric, Metric):
        return deepcopy(metric)
    elif isinstance(metric, str):
        if metric in default_metrics:
            return default_metrics[metric]
        else:
            raise RuntimeError(f"Metric '{metric}' unknown")
    else:
        raise RuntimeError("Metric should be specified as a "
                           "string or a Metric object")


def get_metrics(metrics: MetricsType, outputs) -> dict:
    # If metrics are already in dictionary form
    if isinstance(metrics, dict):
        # If keys match, process each metric and create new dictionary
        if set(metrics.keys()) == set(outputs):
            return {o: get_metric(m) for o, m in metrics.items()}
        else:
            raise RuntimeError("Metric dictionary keys should "
                               "match network outputs")
    # Otherwise, create new dictionay with
    # copy of processed metric for each output
    else:
        return {o: get_metric(metrics) for o in outputs}


def get_numpy_size(data) -> Optional[int]:
    sizes = [d.shape[0] for d in data.values()]
    return sizes[0] if len(set(sizes)) <= 1 else None


def batch_numpy(data, batch_size, size):
    # Determine splits to batch data
    splits = range(batch_size, size + 1, batch_size)

    # Perform split, resulting in {key: split data} dictionary
    data = {k: np.split(v, splits, axis=0)
            for k, v in data.items()}

    # Create list with dictionary for each split
    data_list = [{} for _ in splits]

    # Loop through batches of data
    for k, batches in data.items():
        # Copy batches of data into dictionaries
        for d, b in zip(data_list, batches):
            d[k] = b

    return data_list
