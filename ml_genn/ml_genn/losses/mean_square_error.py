from .loss import Loss


class MeanSquareError(Loss):
    """
    Computes the mean squared error between labels and prediction.
    This needs to be combined with a :class:`ml_genn.readouts.Var` readout for the
    loss

    .. math::

        {\\cal L} = \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\frac{1}{2}  \\int_0^T \\left( x_m(t) - y_m(t) \\right)^2 dt 
    
    where :math:`x_m(t)` is the readout var at time :math:`t` and :math:`y_m(t)` the desired 
    output at time :math:`t`, both in batch :math:`m`.
    """
    @property
    def ground_truth(self) -> str:
        return "timestep_value"
