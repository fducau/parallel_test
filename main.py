'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--fft', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.fft:
    print('Using FFT features')
# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    if args.fft:
        net = ResNeXt29_2x64d(cin=9)
    else:
        net = ResNeXt29_2x64d(cin=3)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def P2R(radii, angles):
    return radii * np.exp(1j*angles)
def R2P(x):       
    return np.abs(x), np.angle(x)

def fft_features(data_tensor, which=['magnitude', 'angle', 'real', 'imag']):
    d_fft = np.fft.fft2(data_tensor.numpy())

    output = [data_tensor]
    if 'mag' in which:
        mag, ang = R2P(d_fft)
        mag_rec = np.real(np.fft.ifft2(P2R(mag, ang * 0.)))
        mag_rec = torch.from_numpy(mag_rec.astype('float32'))
        output.append(mag_rec)
    if 'ang' in which:
        mag, ang = R2P(d_fft)
        ang_rec = np.real(np.fft.ifft2(P2R(np.ones(mag.shape), ang)))
        ang_rec = torch.from_numpy(ang_rec.astype('float32'))
        output.append(ang_rec)
    # Q: USE np.fft.fftshift(d_fft) ?
    if 'real' in which:
        real = np.real(d_fft)
        real_rec = np.real(np.fft.ifft2(np.real(d_fft)))
        real_rec = torch.from_numpy(real_rec.astype('float32'))
        output.append(real_rec)
    if 'imag' in which:
        imag = np.imag(d_fft)
        imag_rec = np.real(np.fft.ifft2(np.imag(d_fft) * 1j))
        imag_rec = torch.from_numpy(imag_rec.astype('float32'))
        output.append(imag_rec)

    
    return torch.cat(output, 1)
    # imag_rec = torch.from_numpy(imag_rec.astype('float32'))
    # inputs = torch.cat([inputs, real_rec, imag_rec], 1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.fft:
            inputs = fft_features(inputs)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.fft:
            inputs = fft_features(inputs)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    state['acc_history'] = state['acc_history'] + [acc]
    if acc > best_acc:
        print('Saving..')
        state['net'] = net.module if use_cuda else net
        state['acc'] = acc
        state['epoch'] = epoch
        best_acc = acc

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7')
    

lr = args.lr
state = {
            'net': net.module if use_cuda else net,
            'acc': 0.,
            'epoch': 0.,
            'acc_history':[]
        }

for epoch in range(start_epoch, start_epoch+150):
    
    if epoch == 50 or epoch == 100:
        lr = lr / 10.
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train(epoch)
    test(epoch)
