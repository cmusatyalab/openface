--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 14.01.2017
-- Time: 10:13
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
a = torch.rand(10, 3, 224, 224)

local net = nn.Sequential()
local SpatialZeroPadding = nn.SpatialZeroPadding
local padding = true
local stride1only = false
local SpatialConvolution = nn.SpatialConvolutionMM
local ReLU = nn.ReLU
local SpatialMaxPooling = nn.SpatialMaxPooling
local net = nn.Sequential()

local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

-- Create tables describing VGG configurations A, B, D, E
local cfg = {}
if modelType == 'A' then
    cfg = { 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' }
elseif modelType == 'B' then
    cfg = { 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' }
elseif modelType == 'D' then
    cfg = { 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' }
elseif modelType == 'E' then
    cfg = { 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' }
else
    error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
end

local features = nn.Sequential()
do
    local iChannels = 3;
    for k, v in ipairs(cfg) do
        if v == 'M' then
            net:add(SpatialMaxPooling(2, 2, 2, 2))
            print(net:forward(a):size())
        else
            local oChannels = v;
            net:add(SpatialConvolution(iChannels, oChannels, 3, 3, 1, 1, 1, 1))
            print(net:forward(a):size())
            net:add(ReLU(true))
            print(net:forward(a):size())
            iChannels = oChannels;
        end
    end
end

net:add(nn.View(512 * 7 * 7))
print(net:forward(a):size())
net:add(nn.Linear(512 * 7 * 7, 4096))
print(net:forward(a):size())
net:add(nn.Dropout(0.5))
print(net:forward(a):size())
net:add(nn.Linear(4096, 4096))
print(net:forward(a):size())
net:add(nn.Dropout(0.5))
print(net:forward(a):size())
net:add(nn.Linear(4096, 128))
print(net:forward(a):size())
net:add(nn.Normalize(2))
print(net:forward(a):size())

