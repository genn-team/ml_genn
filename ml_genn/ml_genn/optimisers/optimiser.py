from abc import ABC
from typing import Any, List, Optional, Tuple
from ..utils.model import CustomUpdateModel

from abc import abstractmethod


class Optimiser(ABC):
    """Base class for all optimisers"""

    @abstractmethod
    def set_step(self, state, genn_cu, step: int):
        """Performs optimiser-specific update to compiled optimier
        object at given training step e.g. recalculating learning rates

        Args:
            genn_cu:    GeNN ``CustomUpdate`` object optimiser has been
                        compiled into
            step:       Training step
        """
        pass

    @abstractmethod
    def create_state(self) -> Any:
        """Returns a new optimiser-specific state object. 
        This should contain copies of any attributes of 
        the optimiser which can be modified at runtime
        """
        pass

    @abstractmethod
    def get_model(self, gradient_ref, var_ref, zero_gradient: bool,
                  clamp_var: Optional[Tuple[float, float]] = None) -> CustomUpdateModel:
        """Gets model described by this optimiser

        Args:
            gradient_ref:   GeNN variable reference for model 
                            to read gradient from
            var_ref:        GeNN variable reference to variable to update
            zero_gradient:  Should gradient be zeroed at the end of the
                            optimiser custom update? This is typically the
                            behaviour we want when batch size is 1 but, 
                            otherwise, gradient_ref points to an intermediate
                            reduced gradient which there is no point in zeroing.
            clamp_var:      Should value of variable being updated be clamped
                            after update?
        """
        pass

    @property
    @abstractmethod
    def checkpoint_var_names(self) -> List[str]:
        """Names of optimiser variables which should be checkpointed"""
        pass
