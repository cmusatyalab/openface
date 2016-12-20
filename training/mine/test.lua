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

-- Ancore embedding batch
x1 = torch.rand(batch, embeddingSize)
-- Positive embedding batch
x2 = torch.rand(batch, embeddingSize)
-- Negativep embedding batch
y = torch.FloatTensor { 1, 0, 1 }

-- Testing the loss function forward and backward
loss = nn.L2LossCriterion()
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward({ x1, x2 }, y), '\n')
gradInput = loss:backward({ x1, x2 }, y)

print(b('gradInput[1]:')); print(gradInput[1])
print(b('gradInput[2]:')); print(gradInput[2])
