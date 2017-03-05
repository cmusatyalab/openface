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
torch.setdefaulttensortype("torch.FloatTensor")
a = torch.rand(10, 3, 28, 28)


local conv = nn.SpatialConvolutionMM
local relu = nn.ReLU
local mp = nn.SpatialMaxPooling
local net = nn.Sequential()

net:add(conv(3, 64, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(64, 64, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(mp(2, 2, 2, 2))
print(net:forward(a):size())
net:add(conv(64, 128, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(128, 128, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(mp(2, 2, 2, 2))
print(net:forward(a):size())
net:add(conv(128, 256, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(mp(2, 2, 2, 2))
print(net:forward(a):size())
net:add(conv(256, 512, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(mp(2, 2, 2, 2))
print(net:forward(a):size())
net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size(),'dsadsa')
net:add(mp(1, 1, 2, 2))
print(net:forward(a):size())

net:add(nn.View(512 * 1 * 1)) --Changed
print(net:forward(a):size())
net:add(nn.Linear(512 * 1 * 1, 4096)) --Changed
print(net:forward(a):size())
net:add(relu(true))
print(net:forward(a):size())
net:add(nn.Linear(4096, 128))
print(net:forward(a):size())
net:add(nn.Normalize(2))
print(net:forward(a):size())

