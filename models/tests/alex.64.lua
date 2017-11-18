--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 11.01.2017
-- Time: 19:41
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'torch'
require 'dpnn'
require 'optim'
require 'image'
require 'torchx'
require 'optim'
require 'xlua'
require 'cunn'
paths.dofile('../../training/torch-TripletEmbedding/TripletEmbedding.lua')
torch.setdefaulttensortype("torch.FloatTensor")
a = torch.rand(10, 3, 64, 64)


local SpatialConvolution = nn.SpatialConvolutionMM --lib[1]
local SpatialMaxPooling = nn.SpatialMaxPooling --lib[2]

local net = nn.Sequential()
print(a:size())
net:add(SpatialConvolution(3, 64, 11, 11, 4, 4, 2, 2)) -- 224 -> 55
print(net:forward(a):size())
net:add(nn.ReLU(true))
net:add(nn.SpatialBatchNormalization(64))
print(net:forward(a):size())
net:add(SpatialMaxPooling(3, 3, 2, 2)) -- 55 ->  27
print(net:forward(a):size())
net:add(SpatialConvolution(64, 192, 5, 5, 1, 1, 2, 2)) --  27 -> 27
print(net:forward(a):size())
net:add(nn.ReLU(true))
net:add(nn.SpatialBatchNormalization(192))
print(net:forward(a):size())
net:add(SpatialMaxPooling(3, 3, 2, 2)) --  27 ->  13
print(net:forward(a):size())
net:add(SpatialConvolution(192, 384, 3, 3, 1, 1, 1, 1)) --  13 ->  13
print(net:forward(a):size())
net:add(nn.ReLU(true))
net:add(nn.SpatialBatchNormalization(384))
print(net:forward(a):size())
net:add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)) --  13 ->  13
print(net:forward(a):size())
net:add(nn.ReLU(true))
net:add(nn.SpatialBatchNormalization(256))
print(net:forward(a):size())
net:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) --  13 ->  13
print(net:forward(a):size())
net:add(nn.ReLU(true))
net:add(nn.SpatialBatchNormalization(256))
print(net:forward(a):size())
net:add(SpatialMaxPooling(3, 3, 2, 2)) -- 13 -> 6
print(net:forward(a):size())
net:add(nn.View(256 * 1 * 1)) --Changed
print(net:forward(a):size())
net:add(nn.Dropout(0.5))
print(net:forward(a):size())
net:add(nn.Linear(256 * 1 * 1, 4096)) --Changed
net:add(nn.ReLU(true))
net:add(nn.BatchNormalization(4096))
print(net:forward(a):size())
net:add(nn.Threshold(0, 1e-6))
print(net:forward(a):size())
net:add(nn.Dropout(0.5))
print(net:forward(a):size())
net:add(nn.Linear(4096, 4096), "net:add(nn.Dropout(0.5))")
net:add(nn.ReLU(true))
net:add(nn.BatchNormalization(4096))
print(net:forward(a):size())
net:add(nn.Threshold(0, 1e-6))
print(net:forward(a):size())
net:add(nn.Linear(4096, 128))
print(net:forward(a):size())
net:add(nn.Normalize(2))
print(net:forward(a):size())


