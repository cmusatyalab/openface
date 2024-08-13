# Copyright 2015-2024 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for Pytorch-based face recognition neural network."""

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, inputSize, reduceSize, outputSize,
                 kernelSize, kernelStride, reduceStride=None,
                 pool=None, activation=None, batchNorm=True, padding=True):
        super().__init__()
        self.branches = []
        if reduceStride is None:
            reduceStride = [1] * len(reduceSize)
        if pool is None:
            pool = nn.MaxPool2d((3, 3), stride=(1, 1))
        if activation is None:
            activation = nn.ReLU()

        # conv branches
        for i in range(len(kernelSize)):
            conv_branch = nn.Sequential()
            # 1x1 conv
            conv_branch.append(nn.Conv2d(inputSize, reduceSize[i], (1, 1), stride=reduceStride[i]))
            if batchNorm:
                conv_branch.append(nn.BatchNorm2d(reduceSize[i]))
            conv_branch.append(activation)
            # nxn conv
            pad = np.floor_divide(kernelSize[i], 2) if padding else (0, 0)
            conv_branch.append(nn.Conv2d(reduceSize[i], outputSize[i], kernelSize[i],
                                         stride=kernelStride[i], padding=pad))
            if batchNorm:
                conv_branch.append(nn.BatchNorm2d(outputSize[i]))
            conv_branch.append(activation)
            self.branches.append(conv_branch)

        # pool branch
        pool_branch = nn.Sequential()
        # pool
        pool_branch.append(pool)
        # 1x1 conv
        i = len(kernelSize)
        if len(reduceSize) > i and reduceSize[i] is not None:
            pool_branch.append(nn.Conv2d(inputSize, reduceSize[i], (1, 1), stride=reduceStride[i]))
            if batchNorm:
                pool_branch.append(nn.BatchNorm2d(reduceSize[i]))
            pool_branch.append(activation)
        self.branches.append(pool_branch)

        # reduce branch
        i = len(kernelSize) + 1
        if len(reduceSize) > i and reduceSize[i] is not None:
            reduce_branch = nn.Sequential()
            reduce_branch.append(nn.Conv2d(inputSize, reduceSize[i], (1, 1), stride=reduceStride[i]))
            if batchNorm:
                reduce_branch.append(nn.BatchNorm2d(reduceSize[i]))
            reduce_branch.append(activation)
            self.branches.append(reduce_branch)

        self.branches = nn.ModuleList(self.branches)

    def forward(self, x: Tensor) -> Tensor:
        branch_out = []
        for branch in self.branches:
            res = branch(x)
            branch_out.append(res)

        # Depth concat with padding
        out_height = max(res.shape[2] for res in branch_out)
        out_width = max(res.shape[3] for res in branch_out)
        for i, res in enumerate(branch_out):
            pad_left = int((out_width - res.shape[3]) // 2)
            pad_right = out_width - res.shape[3] - pad_left
            pad_top = int((out_height - res.shape[2]) // 2)
            pad_bottom = out_height - res.shape[2] - pad_top
            branch_out[i] = F.pad(res, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.)

        out = torch.cat(branch_out, dim=1)
        return out


class OpenFaceNet(nn.Module):
    """
    Usage:
        model = OpenFaceNet()

        # If load on CPU
        model.load_state_dict(torch.load('nn4.small2.v1.pt'))

        # If load on GPU
        model.load_state_dict(torch.load('nn4.small2.v1.pt', map_location='cuda:0')) # Pick the right GPU device number
        model.to(torch.device('cuda'))

        # Loading model for inference only
        model.eval()
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.lrn = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.conv2 = nn.Conv2d(64, 64, (1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(192)
        self.incept3a = Inception(inputSize=192, reduceSize=(96, 16, 32, 64), outputSize=(128, 32),
                                  kernelSize=((3, 3), (5, 5)), kernelStride=((1, 1), (1, 1)),
                                  pool=nn.MaxPool2d((3, 3), stride=(2, 2)))
        self.incept3b = Inception(inputSize=256, reduceSize=(96, 32, 64, 64), outputSize=(128, 64),
                                  kernelSize=((3, 3), (5, 5)), kernelStride=((1, 1), (1, 1)),
                                  pool=nn.LPPool2d(2, (3, 3), stride=(3, 3)))
        self.incept3c = Inception(inputSize=320, reduceSize=(128, 32, None, None), outputSize=(256, 64),
                                  kernelSize=((3, 3), (5, 5)), kernelStride=((2, 2), (2, 2)),
                                  pool=nn.MaxPool2d((3, 3), stride=(2, 2)))
        self.incept4a = Inception(inputSize=640, reduceSize=(96, 32, 128, 256), outputSize=(192, 64),
                                  kernelSize=((3, 3), (5, 5)), kernelStride=((1, 1), (1, 1)),
                                  pool=nn.LPPool2d(2, (3, 3), stride=(3, 3)))
        self.incept4e = Inception(inputSize=640, reduceSize=(160, 64, None, None), outputSize=(256, 128),
                                  kernelSize=((3, 3), (5, 5)), kernelStride=((2, 2), (2, 2)),
                                  pool=nn.MaxPool2d((3, 3), stride=(2, 2)))
        self.incept5a = Inception(inputSize=1024, reduceSize=(96, 96, 256), outputSize=(384,),
                                  kernelSize=((3, 3),), kernelStride=((1, 1),),
                                  pool=nn.LPPool2d(2, (3, 3), stride=(3, 3)))
        self.incept5b = Inception(inputSize=736, reduceSize=(96, 96, 256), outputSize=(384,),
                                  kernelSize=((3, 3),), kernelStride=((1, 1),),
                                  pool=nn.MaxPool2d((3, 3), stride=(2, 2)))
        self.avgpool = nn.AvgPool2d((3, 3), stride=(1, 1))
        self.ln = nn.Linear(736, 128)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)                           # Layer 1
        x = self.bn1(x)                             # Layer 2
        x = self.relu(x)                            # Layer 3
        x = self.maxpool(x)                         # Layer 4
        x = self.lrn(x)                             # Layer 5
        x = self.conv2(x)                           # Layer 6
        x = self.bn2(x)                             # Layer 7
        x = self.relu(x)                            # Layer 8
        x = self.conv3(x)                           # Layer 9
        x = self.bn3(x)                             # Layer 10
        x = self.relu(x)                            # Layer 11
        x = self.lrn(x)                             # Layer 12
        x = self.maxpool(x)                         # Layer 13
        x = self.incept3a(x)                        # Layer 14
        x = self.incept3b(x)                        # Layer 15
        x = self.incept3c(x)                        # Layer 16
        x = self.incept4a(x)                        # Layer 17
        x = self.incept4e(x)                        # Layer 18
        x = self.incept5a(x)                        # Layer 19
        # Reshape to (-1, 736, 3, 3)                # Layer 20
        x = self.incept5b(x)                        # Layer 21
        x = self.avgpool(x)                         # Layer 22
        # Reshape to (-1, 736)                      # Layer 23
        x = x.view((-1, 736))                       # Layer 24
        x = self.ln(x)                              # Layer 25
        x = F.normalize(x, p=2, dim=1, eps=1e-10)   # Layer 26
        return x
