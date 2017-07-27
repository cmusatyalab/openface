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
a = torch.rand(10, 3, 96, 96)


local net = nn.Sequential()
net:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 2, 2, 3, 3))
print(net:forward(a):size())
net:add(nn.SpatialBatchNormalization(64))
print(net:forward(a):size())
net:add(nn.ReLU())
print(net:forward(a):size())

net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
print(net:forward(a):size())
net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
print(net:forward(a):size())

-- Inception (2)
net:add(nn.SpatialConvolutionMM(64, 64, 1, 1))
print(net:forward(a):size())
net:add(nn.SpatialBatchNormalization(64))
print(net:forward(a):size())
net:add(nn.ReLU())
print(net:forward(a):size())
net:add(nn.SpatialConvolutionMM(64, 192, 3, 3, 1, 1, 1))
print(net:forward(a):size())
net:add(nn.SpatialBatchNormalization(192))
print(net:forward(a):size())
net:add(nn.ReLU())
print(net:forward(a):size())

net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
print(net:forward(a):size())
net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
print(net:forward(a):size())

-- Inception (3a)
net:add(nn.Inception {
    inputSize = 192,
    kernelSize = { 3, 5 },
    kernelStride = { 1, 1 },
    outputSize = { 128, 32 },
    reduceSize = { 96, 16, 32, 64 },
    pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (3b)
net:add(nn.Inception {
    inputSize = 256,
    kernelSize = { 3, 5 },
    kernelStride = { 1, 1 },
    outputSize = { 128, 64 },
    reduceSize = { 96, 32, 64, 64 },
    pool = nn.SpatialLPPooling(256, 2, 3, 3, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (3c)
net:add(nn.Inception {
    inputSize = 320,
    kernelSize = { 3, 5 },
    kernelStride = { 2, 2 },
    outputSize = { 256, 64 },
    reduceSize = { 128, 32, nil, nil },
    pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (4a)
net:add(nn.Inception {
    inputSize = 640,
    kernelSize = { 3, 5 },
    kernelStride = { 1, 1 },
    outputSize = { 192, 64 },
    reduceSize = { 96, 32, 128, 256 },
    pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (4b)
net:add(nn.Inception {
    inputSize = 640,
    kernelSize = { 3, 5 },
    kernelStride = { 1, 1 },
    outputSize = { 224, 64 },
    reduceSize = { 112, 32, 128, 224 },
    pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (4e)
net:add(nn.Inception {
    inputSize = 640,
    kernelSize = { 3, 5 },
    kernelStride = { 2, 2 },
    outputSize = { 256, 128 },
    reduceSize = { 160, 64, nil, nil },
    pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (5a)
net:add(nn.Inception {
    inputSize = 1024,
    kernelSize = { 3 },
    kernelStride = { 1 },
    outputSize = { 384 },
    reduceSize = { 192, 128, 384 },
    pool = nn.SpatialLPPooling(960, 2, 3, 3, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

-- Inception (5b)
net:add(nn.Inception {
    inputSize = 896,
    kernelSize = { 3 },
    kernelStride = { 1 },
    outputSize = { 384 },
    reduceSize = { 192, 128, 384 },
    pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
    batchNorm = true
})
print(net:forward(a):size())

net:add(nn.SpatialAveragePooling(3, 3))
print(net:forward(a):size())

-- Validate shape with:
-- net:add(nn.Reshape(896))

net:add(nn.View(896))
print(net:forward(a):size())
net:add(nn.Linear(896, 128))
print(net:forward(a):size())
net:add(nn.Normalize(2))
print(net:forward(a):size())
