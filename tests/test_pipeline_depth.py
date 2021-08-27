from ml_genn.model import Model
from ml_genn.layers import (AvePool2D, AvePool2DConv2D, AvePool2DDense, Conv2D, 
                            Dense, IdentitySynapses, InputLayer, FSReluNeurons, FSReluInputNeurons)

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
    input = InputLayer("input", (224, 224, 3), neurons=FSReluInputNeurons())
    conv0 = Conv2D("conv0", 64, 7, 2, conv_padding='same', neurons=FSReluNeurons())
    pool0 = AvePool2D("pool0", 3, neurons=FSReluNeurons())
    conv0.connect([input])
    pool0.connect([conv0])
    
    # block 1
    block0_conv0 = Conv2D("block0_conv0", 64, 3, conv_padding='same', neurons=FSReluNeurons())
    block0_conv1 = Conv2D("block0_conv1", 64, 3, conv_padding='same', neurons=FSReluNeurons())
    block0_identity = IdentitySynapses()
    block0_conv0.connect([pool0])
    block0_conv1.connect([block0_conv0])
    block0_identity.connect(pool0, block0_conv1)
    
    # block 2
    block1_conv0 = Conv2D("block1_conv0", 64, 3, conv_padding='same', neurons=FSReluNeurons())
    block1_conv1 = Conv2D("block1_conv1", 64, 3, conv_padding='same', neurons=FSReluNeurons())
    block1_identity = IdentitySynapses()
    block1_conv0.connect([block0_conv1])
    block1_conv1.connect([block1_conv0])
    block1_identity.connect(block0_conv1, block1_conv1)
    
    pool1 = AvePool2D("pool1", 56, neurons=FSReluNeurons())
    pool1.connect([block1_conv1])
    
    # create model
    model = Model([input], [pool1], name="test_pipe_resnet")
    print(model.calc_pipeline_depth())
    
#test_pipe_sequential()
test_pipe_resnet()