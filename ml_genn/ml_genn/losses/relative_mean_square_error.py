from typing import Any, Optional
from .loss import Loss


class RelativeMeanSquareError(Loss):
    """
    Computes the mean squared error between prediction of incorrect 
    outputs and correct output when there are two or more label classes, 
    specified as integers.
    
    When combined with a :class:`ml_genn.readouts.Var` readout this implements the 
    relative MSE loss

    .. math::

        {\\cal L} = \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\frac{1}{2} \\sum_{i \\neq l(m)} \\left( t_{m,i} - t_{m,l(m)} - \\Delta\\right)^2 

    where :math:`\\Delta` is a free parameter that indicates the desired temporal distance
    between the spike time of the correct output neuron :math:`t_{m,l(m)}` and the 
    spikes of the other output neurons. 
    This loss is typically used in a classification context. 
    See, e.g. [Goeltz2025]_
    """
    def __init__(self, delta: float, record_key: Optional[Any] = None):
        super().__init__(record_key)
        self.delta = delta

    @property
    def ground_truth(self) -> str:
        return "example_label"
