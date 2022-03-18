import numpy as np

def get_numpy_size(data):
    sizes = [d.shape[0] for d in data.values()]
    return sizes[0] if len(set(sizes)) <= 1 else None

def batch_numpy(data, batch_size, size):
    # Determine splits to batch data
    splits = range(batch_size, size, batch_size)
    
    # Perform split, resulting in {key: split data} dictionary
    data = {k : np.split(v, splits, axis=0)
            for k, v in data.items()}
    
    # Create list with dictionary for each split
    data_list = [{} for _ in splits]
    
    # Loop through batches of data
    for k, batches in data.items():
        # Copy batches of data into dictionaries
        for d, b in zip(data_list, batches):
            d[k] = b
    
    return data_list