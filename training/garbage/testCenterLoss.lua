--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 16/02/2017
-- Time: 09:02
-- To change this template use File | Settings | File Templates.
--

require 'nn'
require 'CenterLoss'
colour = require 'trepl.colorize'
local b = colour.blue


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(0)

batch = 15
embeddingSize = 3

inputs = torch.FloatTensor {
    {
        { 2, 6, 9 }
    },
    {
        { -1, -2, -3 }
    }
}
print(inputs:dim())
labels = torch.FloatTensor { 1,  1 }


loss = nn.CenterLossCriterion(1)
print(loss)
print(colour.red('loss: '), loss:forward(inputs, labels), '\n')
gradInput = loss:backward(inputs, labels)
print(b('gradInput: CenterLossCriterion Loss')); print(gradInput)
