--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 06.03.2017
-- Time: 08:29
-- To change this template use File | Settings | File Templates.
--



cuda = false

require 'nn'
require '../loss/KLDivergence'
torch.setdefaulttensortype('torch.FloatTensor')
if cuda then
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
    cutorch.manualSeedAll(0)
end

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

x =  torch.Tensor { { 0.1, 0.2 }, { 0.3, 0.4 }, { 0.5, 0.6 }, { 0.7, 0.8 }, { 0.9, 0.95 }, { 0.975, 0.99 } }
print(b('ancore embedding batch:')); print(x)

y = torch.Tensor { 1, 1, 1, 2, 2, 2 }
print(b('positive embedding batch:')); print(y)

-- Testing the loss function forward and backward
loss = nn.BatchKLDivCriterion()
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward(x, y), '\n')
gradInput = loss:backward(x, y)
print(b('gradInput[1]:')); print(gradInput[1])

