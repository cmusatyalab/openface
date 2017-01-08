--------------------------------------------------------------------------------
-- Fresh embedding training example
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
-- Positive training samples/images
pImgs = torch.rand(batch, 3, imgSize, imgSize)
-- Negative training samples/images
nImgs = torch.rand(batch, 3, imgSize, imgSize)

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

convNetPos = convNet:clone('weight', 'bias', 'gradWeight', 'gradBias')
convNetNeg = convNet:clone('weight', 'bias', 'gradWeight', 'gradBias')

-- Parallel container
parallel = nn.ParallelTable()
parallel:add(convNet)
parallel:add(convNetPos)
parallel:add(convNetNeg)

print(b('Fresh-embeddings-computation network:')); print(parallel)

-- Cost function
loss = nn.TripletEmbeddingCriterion()

for i = 1, 9 do
   print(colour.green('Epoch ' .. i))
   predict = parallel:forward({aImgs, pImgs, nImgs})
   err = loss:forward(predict)
   errGrad = loss:backward(predict)
   parallel:zeroGradParameters()
   parallel:backward({aImgs, pImgs, nImgs}, errGrad)
   parallel:updateParameters(0.01)

   print(colour.red('loss: '), err)
   print(b('gradInput[1]:')); print(errGrad[1])
end
