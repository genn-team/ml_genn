from typing import Any, Optional
from .connectivity_optimiser import ConnectivityOptimiser


class DeepR(ConnectivityOptimiser):
    """Implementation of Deep-Rewiring (https://arxiv.org/abs/1711.05136) with
    simplifications introduced by Bellec et al. (10.1038/s41467-020-17236-y)
    
    Args:
        l1_strength:            Strength of L1 regularisation to apply
                                before pruning connections
        rewiring_record_key:    Key to store rewiring data in 
                                callback data dictionary"""
    def __init__(self, l1_strength: float = 0.01,
                 rewiring_record_key: Optional[Any] = None):
        self.rewiring_record_key = rewiring_record_key
        self.l1_strength = l1_strength