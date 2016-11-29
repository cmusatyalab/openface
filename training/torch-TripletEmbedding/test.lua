--------------------------------------------------------------------------------
-- Test function for TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------

cuda = false

require 'nn'
require 'TripletEmbedding'
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
print(b('ancore embedding batch:')); print(a)
-- Positive embedding batch
p = torch.rand(batch, embeddingSize)
print(b('positive embedding batch:')); print(p)
-- Negativep embedding batch
n = torch.rand(batch, embeddingSize)
print(b('negative embedding batch:')); print(n)

-- Testing the loss function forward and backward
loss = nn.TripletEmbeddingCriterion(.2)
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward({a, p, n}), '\n')
gradInput = loss:backward({a, p, n})
print(b('gradInput[1]:')); print(gradInput[1])
print(b('gradInput[2]:')); print(gradInput[2])
print(b('gradInput[3]:')); print(gradInput[3])

-- Jacobian test
d = 1e-6
jacobian = {}
zz = torch.Tensor{
   {1, 0, 0},
   {0, 1, 0},
   {0, 0, 1},
}

for k = 1, 3 do
   jacobian[k] = torch.zeros(a:size())
   z = zz[k]
   for i = 1, a:size(1) do
      for j = 1, a:size(2) do
         pert = torch.zeros(a:size())
         pert[i][j] = d
         outA = loss:forward({a - pert*z[1], p - pert*z[2], n - pert*z[3]})
         outB = loss:forward({a + pert*z[1], p + pert*z[2], n + pert*z[3]})
         jacobian[k][i][j] = (outB - outA)/(2*d)
      end
   end
end

print(b('jacobian[1]:')); print(jacobian[1])
print(b('jacobian[2]:')); print(jacobian[2])
print(b('jacobian[3]:')); print(jacobian[3])
