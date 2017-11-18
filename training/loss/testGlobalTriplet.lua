cuda = false

require 'nn'
require 'GlobalTriplet'
require '../torch-TripletEmbedding/TripletEmbedding'
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
a = torch.rand(batch, embeddingSize)
-- Positive embedding batch
p = torch.rand(batch, embeddingSize)
-- Negativep embedding batch
n = torch.rand(batch, embeddingSize)

-- Testing the loss function forward and backward
loss = nn.TripletPlusGlobalCriterion()
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward({ a, p, n }), '\n')
gradInput = loss:backward({ a, p, n })
