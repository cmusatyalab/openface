--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 06.03.2017
-- Time: 08:29
-- To change this template use File | Settings | File Templates.
--

require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

nsize = 8
xsize = 4

y = torch.Tensor { 1, 1, 2, 2 }
y1=torch.Tensor{}
print(y,y1)
x = nn.Normalize(2):forward(torch.rand(xsize, nsize))
print(x)
loss = nn.CrossEntropyCriterion()
print(colour.red('loss: '), loss:forward(x, y), '\n')
gradInput = loss:backward(x, y)
print(gradInput)
print(y,y1)