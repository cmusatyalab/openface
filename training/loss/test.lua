--------------------------------------------------------------------------------
-- Test function for TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------
require('mobdebug').start()
cuda = false

require 'nn'
require 'MeanLoss'
require 'CenterLoss'
require 'ConsLoss'
require 'MinDiffLoss'
require 'L2HingeLoss'
colour = require 'trepl.colorize'
local b = colour.blue
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(0)

batch = 3
embeddingSize = 2

a = torch.rand(batch, embeddingSize)
p = torch.rand(batch, embeddingSize)

y = torch.FloatTensor {{ 1, 1, -1 }}

loss = nn.L2HingeEmbeddingCriterion()
print(loss)
print(colour.red('loss: '), loss:forward({ a, p }, y), '\n')
gradInput = loss:backward(a)
print(b('gradInput: Hinge Loss')); print(gradInput)

--
--
--print(b('embedding batch:')); print(a)
--
--- - Testing the loss function forward and backward
---- loss = nn.MeanLossCriterion(.2, 3)
---- print(colour.red('loss: '), loss:forward(a), '\n')
---- gradInput = loss:backward(a)
---- print(b('gradInput: Mean Loss')); print(gradInput)
----
----
-- loss = nn.ConsLossCriterion(.2, 2)
-- print(colour.red('loss: '), loss:forward(a), '\n')
-- gradInput = loss:backward(a)
-- print(b('gradInput: Center Loss')); print(gradInput)
--
-- a = torch.randn(batch, embeddingSize)
--
--
-- print(b('embedding batch:')); print(a)
--
-- loss = nn.ConsLossCriterion(.2, 2)
-- print(colour.red('loss: '), loss:forward(a), '\n')
-- gradInput = loss:backward(a)
-- print(b('gradInput: Center Loss')); print(gradInput)
--
----
---- loss = nn.MinDiffLossCriterion(.2, 3)
---- print(colour.red('loss: '), loss:forward(a), '\n')
---- gradInput = loss:backward(a)
----print(b('gradInput: Min Diff Loss')); print(gradInput)