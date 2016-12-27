--------------------------------------------------------------------------------
-- Test function for TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------

cuda = false

require 'nn'
require 'L2Loss'
torch.setdefaulttensortype('torch.FloatTensor')
if cuda then
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
    cutorch.manualSeedAll(0)
end

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

batch = 3
embeddingSize = 5

x = torch.FloatTensor { { 0.1, .2, .5 }, { -.1, -.2, -.3 }, { .2, .3, .4 }, { .3, .4, .5 } }

x1 = x[{ { 1, 2 } }]
x2 = x[{ { 3, 4 } }]
y = torch.FloatTensor { 1, 0 }

--x = nn.Normalize(2):forward(x)

x1 = x[{ { 1, 2 } }]
-- Positive embedding batch
x2 = x[{ { 3, 4 } }]
-- Negativep embedding batch
y = torch.FloatTensor { 1, -1 }
print(colour.red('Y: '), y, '\n')

print(colour.red('X1: '), x1, '\n')
print(colour.red('X2: '), x2, '\n')
print(x1, x2)
-- Testing the loss function forward and backward
loss = nn.L2LossCriterion()
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward({ x1, x2 }, y), '\n')
print(colour.red('Y: '), y, '\n')
gradInput = loss:backward({ x1, x2 }, y)

print(b('gradInput[1]:')); print(gradInput[1])
print(b('gradInput[2]:')); print(gradInput[2])
