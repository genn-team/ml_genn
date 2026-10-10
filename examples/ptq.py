#!/usr/bin/env python
# coding: utf-8
# Author: Sirine Arfa
# Created: 12.09.2023
# Description: Training SNN with Quantization Aware training for the DVS gesture task

# ===============================
# Imports
# ===============================

# Tonic imports
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset

# SNNtorch imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

# Other imports
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import numpy as np

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ===============================
# Dataset Loading
# ===============================

batch_size = 16
sensor_size = (32, 32, 2)

train_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.Downsample(spatial_factor=0.25),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1e3)
])

test_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.Downsample(spatial_factor=0.25),
    transforms.ToFrame(sensor_size=sensor_size, time_window=1e3)
])

trainset = tonic.datasets.DVSGesture(save_to='./data', transform=train_transform, train=True)
testset = tonic.datasets.DVSGesture(save_to='./data', transform=test_transform, train=False)

cached_trainset = DiskCachedDataset(trainset, cache_path='./data/cache/dvs/train')
cached_testset = DiskCachedDataset(testset, cache_path='./data/cache/dvs/test')

train_loader = DataLoader(
    cached_trainset, shuffle=True, batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
)
test_loader = DataLoader(
    cached_testset, shuffle=True, batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False)
)

# ===============================
# Model Definition
# ===============================

num_classes = 11
slope = 9.70
spike_grad = surrogate.fast_sigmoid(slope)
beta = 0.93  # Decay rate parameter

net = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=1, bias=False),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.LPPool2d(1, kernel_size=(2, 2)),
    nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.LPPool2d(1, kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8 * 3 * 3, 256, bias=False),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, num_classes, bias=False),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)

# Forward pass function
def forward(net, data):
    spk_rec = []
    utils.reset(net)  # Reset hidden states
    for step in range(data.size(0)):
        spk_out, _ = net(data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

# ===============================
# Training Techniques
# ===============================

optimizer = torch.optim.Adam(net.parameters(), lr=2.4e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
loss_dependent = False
weight_clip = True
grad_clip = True
early_stopping = True
patience = 500

# EarlyStopping class
class EarlyStoppingAcc:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_loss_min = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, test_loss, model):
        score = test_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"Early Stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
        else:
            self.best_score = score
            self.save_checkpoint(test_loss, model)
            self.counter = 0

    def save_checkpoint(self, test_loss, model):
        if self.verbose:
            self.trace_func(
                f"Test Accuracy improved ({self.test_loss_min:.6f} --> {test_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.test_loss_min = test_loss

# ===============================
# Training, Validation, and Testing Functions
# ===============================

# Training function
def train(net, train_loader, loss_fn, optimizer, device, grad_clip=False, weight_clip=False):
    correct, total = 0, 0
    net.train()
    train_loss_minibatch, train_lr_minibatch = [], []
    running_loss = 0
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        data = torch.clamp(data, 0, 1)
        spk_rec = forward(net, data)
        loss = loss_fn(spk_rec, targets)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if weight_clip:
            with torch.no_grad():
                for param in net.parameters():
                    param.clamp_(-1, 1)
        optimizer.step()
        train_loss_minibatch.append(loss.item())
        train_lr_minibatch.append(optimizer.param_groups[0]["lr"])
        accuracy = SF.accuracy_rate(spk_rec, targets)
        total += spk_rec.size(1)
        correct += accuracy * spk_rec.size(1)
    return 100 * correct / total, train_lr_minibatch, train_loss_minibatch, running_loss / len(train_loader)

# Validation function
def validation(net, validation_loader, device):
    correct, total = 0, 0
    running_loss = 0
    validation_loss_minibatch = []
    with torch.no_grad():
        net.eval()
        for test_data, test_targets in validation_loader:
            test_data, test_targets = test_data.to(device), test_targets.to(device)
            test_data = torch.clamp(test_data, 0, 1)
            spk_rec = forward(net, test_data)
            loss = loss_fn(spk_rec, test_targets)
            validation_loss_minibatch.append(loss.item())
            running_loss += loss.item()
            accuracy = SF.accuracy_rate(spk_rec, test_targets)
            total += spk_rec.size(1)
            correct += accuracy * spk_rec.size(1)
    return 100 * correct / total, validation_loss_minibatch, running_loss / len(validation_loader)

# Testing function
def test(net, test_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        net.eval()
        for test_data, test_targets in test_loader:
            test_data, test_targets = test_data.to(device), test_targets.to(device)
            spk_rec = forward(net, test_data)
            correct += SF.accuracy_rate(spk_rec, test_targets) * spk_rec.size(1)
            total += spk_rec.size(1)
    return (correct / total) * 100

# ===============================
# Training Loop
# ===============================

model_name = "PTQ_time_window-1ms-snntorch_dvsgesture_model.pth"
early_stopping_instance = EarlyStoppingAcc(patience=patience, verbose=True, path=model_name)

num_epochs = 100
train_acc_epoch_hist, validation_acc_epoch_hist = [], []
train_loss_epoch_hist, validation_loss_epoch_hist = [], []
best_train_acc, best_val_acc = 0, 0

for epoch in range(num_epochs):
    train_acc_epoch, train_lr_minibatch, train_loss_minibatch, train_loss_epoch = train(
        net, train_loader, loss_fn, optimizer, device, grad_clip, weight_clip
    )
    train_acc_epoch_hist.append(train_acc_epoch)
    train_loss_epoch_hist.append(train_loss_epoch)
    
    val_acc_epoch, validation_loss_minibatch, validation_loss_epoch = validation(net, test_loader, device)
    validation_acc_epoch_hist.append(val_acc_epoch)
    validation_loss_epoch_hist.append(validation_loss_epoch)
    
    if train_acc_epoch > best_train_acc:
        best_train_acc = train_acc_epoch
    if val_acc_epoch > best_val_acc:
        best_val_acc = val_acc_epoch
    
    print(f"Epoch {epoch+1}: Train Acc: {train_acc_epoch:.2f}%, Val Acc: {val_acc_epoch:.2f}%")
    
    if early_stopping:
        early_stopping_instance(val_acc_epoch, net)
        if early_stopping_instance.early_stop:
            print("Early stopping triggered.")
            break

print(f"Best Train Acc: {best_train_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%")
