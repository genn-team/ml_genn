from .loss import Loss


class SparseCategoricalCrossentropy(Loss):
    """
    Computes the crossentropy between labels and prediction 
    when there are two or more label classes, specified as integers.
    
    This enables the following loss functions:
    If combined with a :class:`ml_genn.readouts.SumVar` or :class:`ml_genn.readouts.AvgVar` readout and per_timestep_loss=True:

    .. math::

        {\\cal L} = -\\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\int_0^T \\log \\left( \\frac{\\exp\\left(x_{l(m)}^m(t)\\right)}{\\sum_{k=1}^{N_{\\text{class}}} \\exp\\left(x_{k}^m(t) \\right)} \\right) dt

    where :math:`x` is the readout variable. This is a *per-timestep loss function*.

    If combined with :class:`ml_genn.readouts.SumVar`, :class:`ml_genn.readouts.AvgVar` or :class:`ml_genn.readouts.AvgVarExpWeight` readouts and per_timestep_loss=False:

    .. math::

        {\\mathcal L_{\\text{sum}}} = - \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\log \\left( \\frac{\\exp\\left(\\int_0^T f(x_{l(m)}^m(t)) dt\\right)}{\\sum_{k=1}^{N_{\\text{out}}} \\exp\\left(\\int_0^T f(x_{k}^m(t)) dt\\right)} \\right)

    where :math:`x` is the readout variable and :math:`f(\cdot)` is 
   
    for :class:`ml_genn.readouts.SumVar`: f(x)= x

    for :class:`ml_genn.readouts.AvgVar`: f(x)= x/T

    for :class:`ml_genn.readouts.AvgVarExpWeight`: f(x)= exp(-t/T)*x/T

    If combined with :class:`ml_genn.readouts.EndVar`, and per_timestep_loss=False:

    .. math::

        {\\mathcal L_{\\text{end}}} = - \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\log \\left( \\frac{\\exp\\left(x_{l(m)}^m(T)\\right)}{\\sum_{k=1}^{N_{\\text{out}}} \\exp\\left(x_{k}^m(T)\\right)} \\right)

    where :math:`x_k^m(T)` is the readout variable at time :math:`T`.


    If combined with a :class:`ml_genn.readouts.FirstSpikeTime` readout:

    .. math::

        {\\mathcal L_{\\text{ttfs}}} = - \\frac{1}{N_{\\text{batch}}} \\sum_{m=1}^{N_{\\text{batch}}} \\log \\left( \\frac{\\exp\\left(\\frac{-t_{m,l(m)}}{\tau_0}\\right)}{\\sum_{k=1}^{N_{\\text{out}}} \\exp\\left(\\frac{-t_{m,k}}{\tau_0}\\right)} \\right) + \\alpha \\frac{1}{1.01\\cdot T - t_{m,l(m)}}

    where :math:`t_{m,k}` is the first spike time in neuron :math:`k` in batch :math:`m`.
    """
    @property
    def ground_truth(self) -> str:
        return "example_label"
