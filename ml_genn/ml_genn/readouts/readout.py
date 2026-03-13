import numpy as np

from abc import ABC
from numbers import Number
from typing import List, Tuple
from ..utils.model import NeuronModel

from abc import abstractmethod


class Readout(ABC):
    """Base class for all readouts"""
    
    @abstractmethod
    def add_readout_logic(self, model: NeuronModel, **kwargs):
        """Add any additional state to neuron model
        and functionality required to implement this readout added.

        Args:
            model:  Base neuron model
        """
        pass

    @abstractmethod
    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        """Read out the value from the state of a compiled neuron group.
        
        Args:
            genn_pop:   GeNN ``NeuronGroup`` object population
                        with readout has been compiled into
            batch_size: Batch size of model readout is part of
            shape:      Shape of population
        """
        pass

    @property
    def reset_vars(self) -> List[Tuple[str, str, Number]]:
        """Get list of tuples describing name, type and value
        to reset any state variables added by readout to
        """
        return []


class TimeWindowReadout(Readout):
    """Base class of readouts that allow being constrained to a time window"""
    def __init__(self, window_start=None, window_end=None):
        """Allow to define a window in which to calculate the readout. 
        If no window is defined, default to using the whole trial.
        Window is in units of time (not timesteps)."""
        self.window_start = window_start
        self.window_end = window_end

    def window_start_end(self, **kwargs):
        window_start = self.window_start or 0
        window_end = self.window_end or kwargs["example_timesteps"]*kwargs["dt"]
        return window_start, window_end
    
    def windowed_readout_code(self, code: str, **kwargs):
        if self.window_start is not None or self.window_end is not None:
            window_start, window_end = self.window_start_end(**kwargs)
            return f"""
                if (t >= {window_start} && t < {window_end}) {{
                    {code}
                }}
                """
        else:
            return code

    def back_windowed_readout_code(self, code: str, **kwargs):
        if self.window_start is not None or self.window_end is not None:
            window_start, window_end = self.window_start_end(**kwargs)
            T = kwargs["dt"] * kwargs["example_timesteps"]
            return f"""
                if (t <= {T-window_start} && t > {T-window_end}) {{
                    {code}
                }}
                """
        else:
            return code
