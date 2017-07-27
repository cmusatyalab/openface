--------------------------------------------------------------------------------
-- Recycling embedding training example
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr 15
--------------------------------------------------------------------------------

package.path = "../?.lua;" .. package.path

require 'nn'
require 'TripletEmbedding'
colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

batch = 5
embeddingSize = 3
imgSize = 20

-- Ancore training samples/images
aImgs = torch.rand(batch, 3, imgSize, imgSize)
-- Positive embedding batch
p = torch.rand(batch, embeddingSize)
-- Negativep embedding batch
n = torch.rand(batch, embeddingSize)

-- Network definition
convNet = nn.Sequential()
convNet:add(nn.SpatialConvolution(3, 8, 5, 5))
convNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
convNet:add(nn.ReLU())
convNet:add(nn.SpatialConvolution(8, 8, 5, 5))
convNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
convNet:add(nn.ReLU())
convNet:add(nn.View(8*2*2))
convNet:add(nn.Linear(8*2*2, embeddingSize))
convNet:add(nn.BatchNormalization(0))

-- Parallel container
parallel = nn.ParallelTable()
parallel:add(convNet)
parallel:add(nn.Identity())
parallel:add(nn.Identity())

print(b('Recycling-previous-epoch-embeddings network:')); print(parallel)

-- Cost function
loss = nn.TripletEmbeddingCriterion()

for i = 1, 4 do
   print(colour.green('Epoch ' .. i))
   predict = parallel:forward({aImgs, p, n})
   err = loss:forward(predict)
   errGrad = loss:backward(predict)
   parallel:zeroGradParameters()
   parallel:backward({aImgs, p, n}, errGrad)
   parallel:updateParameters(0.01)

   print(colour.red('loss: '), err)
   print(b('gradInput[1]:')); print(errGrad[1])
end
