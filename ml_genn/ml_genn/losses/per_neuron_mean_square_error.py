from .loss import Loss


class PerNeuronMeanSquareError(Loss):
    """Computes the mean squared error between labels and prediction.

    It should be combined with a :class:`ml_genn.readouts.FirstSpikeTime` readout for the loss

    .. math::
        {\\cal L} = \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\frac{1}{2}  \sum_i^N \\left( t_{m,i} - \hat{t}_{m,i} \\right)^2
        

    where :math:`t_{m,i}` is the first spike time of the output neuron :math:`i` at time 
    :math:`t` and :math:`\hat{t}_{m,i}` the desired spike time in output :math:`i`, 
    both in batch :math:`m`.
    """
    @property
    def ground_truth(self) -> str:
        return "example_value"

