--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 06.03.2017
-- Time: 08:29
-- To change this template use File | Settings | File Templates.
--



cuda = false

require 'nn'
require '../loss/LSSS'

torch.setdefaulttensortype('torch.FloatTensor')
if cuda then
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
    cutorch.manualSeedAll(0)
end

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)
opt = {}
opt.cuda = True
nsize = 10
xsize = 8
input = torch.Tensor { { 19, 9 }, { 15, 7 }, { 7, 2 }, { 17, 6 } }
input = torch.randn(xsize, nsize)
target = torch.Tensor { 1, 1, 2, 2 , 3, 3, 4, 4 }

--input = nn.Normalize(2):forward(input)
print(input)
loss = nn.LiftedStructuredSimilaritySoftmaxCriterion()
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward(input, target), '\n')
gradInput = loss:backward(input, target)
print(gradInput)
print(colour.red('loss: '), loss:forward(input, target), '\n')
gradInput = loss:backward(input, target)
print(gradInput)

-- 19   9
-- 15   7
--  7   2
-- 17   6
--[torch.FloatTensor of size 4x2]
--  3.4632
--  3.4632
--  9.7614
-- 9.7614
--loss: 	26.819639205933
--
-- 1.2941 -0.0673
-- -0.1578 -0.6960
-- -5.4700  1.2442
-- 0.8008  2.1873
--[torch.FloatTensor of size 4x2]