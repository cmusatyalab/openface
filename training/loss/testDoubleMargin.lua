--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 06.03.2017
-- Time: 08:29
-- To change this template use File | Settings | File Templates.
--



cuda = false

require 'nn'
require '../loss/DoubleMargin'

torch.setdefaulttensortype('torch.FloatTensor')
if cuda then
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
    cutorch.manualSeedAll(0)
end

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

x = torch.rand(100, 128)

y = torch.rand(100)

function findClassId(i, numberPerClass)
    local id_ = 1
    local ind = math.floor((i - 1) / numberPerClass)
    id_ = id_ + ind
    return id_
end

local function pairss(embeddings, numPerClass)
    local a1s_table = {}
    local a2s_table = {}
    local targets_table = {}
    local mapper = {}
    for i = 1, embeddings:size(1) do
        local classIdi = findClassId(i, numPerClass)
        for j = 1, embeddings:size(1) do
            if i ~= j then
                local classIdj = findClassId(j, numPerClass)
                table.insert(a1s_table, embeddings[i])
                table.insert(a2s_table, embeddings[j])

                local target
                if classIdi == classIdj then
                    target = 1
                else
                    target = 0
                end
                table.insert(targets_table, target)
                table.insert(mapper, { i, j })
            end
        end
    end
    local a1s = torch.cat(a1s_table):view(table.getn(a1s_table), 128)
    local a2s = torch.cat(a2s_table):view(table.getn(a2s_table), 128)

    local targets = torch.Tensor(targets_table)


    local as = { a1s, a2s }

    return as, targets, mapper
end


x = nn.Normalize(2):forward(x)

local ass, targets, mapper = pairss(x, 100)
loss = nn.DoubleMarginCriterion()
if cuda then loss = loss:cuda() end
print(colour.red('loss: '), loss:forward(ass, targets), '\n')
gradInput = loss:backward(ass, targets)
print(colour.red('loss: '), loss:forward(ass, targets), '\n')
gradInput = loss:backward(ass, targets)
local gradient_all = torch.Tensor(x:size(1), ass[1]:size(2)):type(x:type())
print('GradInput', gradInput[1], gradInput[2])
for i = 1, table.getn(mapper) do
    gradient_all[mapper[i][1]]:add(gradInput[1][i])
    gradient_all[mapper[i][2]]:add(gradInput[2][i])
end
print('GradAll', gradient_all)