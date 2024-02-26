import math
import numpy as np

def find_signed_scale(data, num_bits: int, percentile: float):
    # Split data into positive and negative
    positive_mask = (data > 0)
    positive_data = data[positive_mask]
    negative_data = data[np.logical_not(positive_mask)]

    # Calculate desired percentile
    positive_perc = np.percentile(positive_data, percentile)
    negative_perc = np.percentile(-negative_data, percentile)

    # Calculate the largest of these
    max_val = max(positive_perc, negative_perc)
    
    # Calculate high bit and low bit
    # **NOTE** we floor so max is 2**(high_bit + 1) - 1
    # **NOTE** one bit is used for sign
    high_bit =  math.floor(math.log(max_val, 2))
    low_bit = high_bit - (num_bits - 2)
    
    # We scale to multiples of the low bit
    scale = (2.0 ** low_bit)
    
    # Calculate min and max
    min_quant = (-2.0 ** (high_bit + 1))
    max_quant = (2.0 ** (high_bit + 1)) - scale

    # Return range and scale
    return min_quant, max_quant, scale

def find_unsigned_scale(data, num_bits: int, percentile: float):
    # Calculate desired percentile
    perc = np.percentile(data, percentile)
    
    # Calculate high bit and low bit
    # **NOTE** we floor so max is 2**(high_bit + 1) - 1
    high_bit = math.floor(math.log(perc, 2))
    low_bit = high_bit - (num_bits - 1)

    # We scale to multiples of the low bit
    scale = (2.0 ** low_bit)
    
    # Calculate max
    max_quant = (2.0 ** (high_bit + 1)) - scale
    
    # Return range and scale
    return 0.0, max_quant, scale

def quantise_signed(data, num_bits: int, percentile: float):
    # Find scaling factors
    min_quant, max_quant, scale = find_signed_scale(data, num_bits,
                                                    percentile)

    # Quantise, clip and return
    return np.clip(scale * np.round(data / scale), min_quant, max_quant)


def quantise_unsigned(data, num_bits: int, percentile: float):
    # Find scaling factors
    min_quant, max_quant, scale = find_unsigned_scale(data, num_bits,
                                                      percentile)

    # Quantise, clip and return
    return np.clip(scale * np.round(data / scale), min_quant, max_quant)