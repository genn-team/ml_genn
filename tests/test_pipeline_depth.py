from ml_genn.model import Model
from ml_genn.layers import (AvePool2DConv2D, AvePool2DDense, Conv2D, Dense,
                            InputLayer, FSReluNeurons, FSReluInputNeurons)

def test_pipe_sequential():
    input = InputLayer("input", (28, 28, 1), neurons=FSReluInputNeurons())
    conv0 = Conv2D("conv0", 16, 5, neurons=FSReluNeurons())
    conv1 = AvePool2DConv2D("conv1", 8, 2, 5, neurons=FSReluNeurons())
    dense0 = AvePool2DDense("dense0", 128, 2, neurons=FSReluNeurons())
    dense1 = Dense("dense1", 64, neurons=FSReluNeurons())
    output = Dense("output", 10, neurons=FSReluNeurons())
    
    conv0.connect([input])
    conv1.connect([conv0])
    dense0.connect([conv1])
    dense1.connect([dense0])
    output.connect([dense1])
    
    # create model
    model = Model([input], [output], name="test_pipe_sequential")

    # Check that pipeline depth is correct
    assert model.calc_pipeline_depth() == 4

def test_pipe_resnet():
    pass

test_pipe_sequential()
test_pipe_resnet()