# Tonic imports
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from tonic.slicers import SliceByTime
from tonic import SlicedDataset

# SNNtorch imports
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

# Other imports
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import statistics
import itertools
from collections import defaultdict

# Device setup
device = torch.device("cpu")#torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ===============================
# Dataset Loading
# ===============================

batch_size = 16
sensor_size = (32, 32, 2)

test_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.Downsample(spatial_factor=0.25),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1e3)
])

testset = tonic.datasets.DVSGesture(save_to='./data', transform=test_transform, train=False)

cached_testset = DiskCachedDataset(testset, cache_path='./data/cache/dvs/test')

test_loader = DataLoader(
    cached_testset, shuffle=False, batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
)

# ===============================
# Model Definition
# ===============================

num_classes = 11
slope = 9.70
beta = 0.93  # Decay rate parameter
record = True

net = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=1, bias=False),
    snn.Leaky(beta=beta, init_hidden=True),
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
    snn.Leaky(beta=beta, init_hidden=True),
    nn.LPPool2d(1, kernel_size=(2, 2)),
    nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
    snn.Leaky(beta=beta, init_hidden=True),
    nn.LPPool2d(1, kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8 * 3 * 3, 256, bias=False),
    snn.Leaky(beta=beta, init_hidden=True),
    nn.Linear(256, num_classes, bias=False),
    snn.Leaky(beta=beta, init_hidden=True, output=True)
).to(device)
#print([n for n, _ in net.named_children()])

def get_output(output):
    if isinstance(output, torch.Tensor):
        return (output.detach().cpu(),)
    else:
        return (o.detach().cpu() for o in output)

if record:
    activations_dict = defaultdict(list)
    for name, mod in list(net.named_modules())[1:-1]:
        print(f"registering forward hook for {name}")
        mod.register_forward_hook(
            lambda m, i, o: activations_dict[name].append(get_output(o)))



#net.

# Forward pass function
def forward(net, data):
    spk_rec = []
    utils.reset(net)  # Reset hidden states
    for step in range(data.size(0)):
        spk_out, _ = net(data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

# ===============================
# Training, Validation, and Testing Functions
# ===============================

# Testing function
def test(net, test_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        net.eval()
        #for test_data, test_targets in test_loader:
        test_data, test_targets = next(iter(test_loader))
        test_data, test_targets = test_data.to(device), test_targets.to(device)
        spk_rec = forward(net, test_data)
        correct += SF.accuracy_rate(spk_rec, test_targets) * spk_rec.size(1)
        total += spk_rec.size(1)
    return (correct / total) * 100

net.load_state_dict(torch.load("PTQ_time_window-1ms-snntorch_dvsgesture_model.pth", 
                               map_location=device, weights_only=True))

print(test(net, test_loader, device))

if record:
    for name, outputs in activations_dict.items():
        print(name)
        for step_output in outputs:
            for o in step_output:
                print(f"\t{o.shape}")
    