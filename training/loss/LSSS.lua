local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')
require 'torchx'
function LiftedStructuredSimilaritySoftmaxCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function LiftedStructuredSimilaritySoftmaxCriterion:makePairs(input, target)
    local a1s_table = {}
    local a2s_table = {}
    local mapper = {}
    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
            if i ~= j and target[i] == target[j] then

                table.insert(a1s_table, input[i])
                table.insert(a2s_table, input[j])
                table.insert(mapper, { i, j })
            end
        end
    end

    local a1s = torch.concat(a1s_table):view(table.getn(a1s_table), input:size(2))
    local a2s = torch.concat(a2s_table):view(table.getn(a2s_table), input:size(2))

    local as
    if opt.cuda then
        as = { a1s:cuda(), a2s:cuda() }
    else
        as = { a1s, a2s }
    end

    return as, mapper
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateOutput(input, target)

    self.Li = torch.Tensor(input:size(1), 1):zero():type(torch.type(input))
    self.dissValues = {}
    self.indices = {}
    self.counter = 0
    self.x1SubX3 = {}
    self.x1SubX2 = {}
    self.x2SubX3 = {}

    self.posPairs, self.mapper = self:makePairs(input, target)
    local x1 = self.posPairs[1]
    local x2 = self.posPairs[2]
    local posPairsCount = x1:size(1)

    self.x1SubX2 = x1 - x2
    self.normX1SubX2 = torch.norm(self.x1SubX2, 2, 2)
    self.expX1SubX2 = torch.exp(self.alpha - self.normX1SubX2)

    for ind = 1, table.getn(self.mapper) do
        local i = self.mapper[ind][1]
        local j = self.mapper[ind][2]
        self.x1SubX3[i] = {}
        self.x2SubX3[i] = {}

        if not self.dissValues[target[i]] then

            self.indices[target[i]] = torch.find((target - target[i]):ne(0):type(torch.type(torch.FloatTensor())), 1)
            self.dissValues[target[i]] = input:index(1, torch.LongTensor(self.indices[target[i]]))
        end
        local dissValue = self.dissValues[target[i]]

        self.x1SubX3[i][1] = dissValue - input[i]:repeatTensor(dissValue:size(1), 1) --x1SubX3
        self.x1SubX3[i][2] = torch.norm(self.x1SubX3[i][1], 2, 2) --normX1SubX3
        self.x1SubX3[i][3] = torch.exp(self.alpha - self.x1SubX3[i][2]) --expX1SubX3

        self.x2SubX3[i][j] = {}
        self.x2SubX3[i][j][1] = dissValue - input[j]:repeatTensor(dissValue:size(1), 1) --x2SubX3
        self.x2SubX3[i][j][2] = torch.norm(self.x2SubX3[i][j][1], 2, 2) --normX2SubX3
        self.x2SubX3[i][j][3] = torch.exp(self.alpha - self.x2SubX3[i][j][2]) --expX2SubX3


        self.counter = self.x2SubX3[i][j][3]:size(1) * 2
        self.Li[i] = self.normX1SubX2[ind] + torch.log(torch.sum(self.x1SubX3[i][3]) + torch.sum(self.x2SubX3[i][j][3]))
    end

    self.output = torch.pow(self.Li, 2):sum() / (3 * posPairsCount)
    return self.output
end

--TODO
function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput(input, target)
    self.gradInput = torch.Tensor(input:size()):zero():type(torch.type(input))
    aa = self:updateGradInput1(input, target)
    self.gradK = {}

    local diffX1X2 = torch.cdiv(self.x1SubX2, self.normX1SubX2:repeatTensor(1, self.x1SubX2:size(2)))
    local dividing = torch.exp(self.Li - self.normX1SubX2)

    for ind = 1, table.getn(self.mapper) do

        local i = self.mapper[ind][1]
        local j = self.mapper[ind][2]

        local x1SubX3 = self.x1SubX3[i][1]
        local normX1SubX3 = self.x1SubX3[i][2]

        if not self.gradK[target[i]] then
            self.gradK[target[i]] = {}
            local dividedIK = -self.x1SubX3[i][3]
            local diffX1X3 = torch.cdiv(x1SubX3, normX1SubX3:repeatTensor(1, x1SubX3:size(2)))
            local x2SubX3 = self.x2SubX3[i][j][1]
            local normX2SubX3 = self.x2SubX3[i][j][2]
            local diffX2X3 = torch.cdiv(x2SubX3, normX2SubX3:repeatTensor(1, x2SubX3:size(2)))
            local dividedJK = -self.x2SubX3[i][j][3]

            local firstDivIK = dividedIK / dividing[ind][1]
            local firstDivIJ = dividedJK / dividing[ind][1]
            local diffIK = torch.cmul(diffX1X3, firstDivIK:repeatTensor(1, diffX1X3:size(2)))
            local diffJK = torch.cmul(diffX2X3, firstDivIJ:repeatTensor(1, diffX2X3:size(2)))
            self.gradK[target[i]][1] = diffIK
            self.gradK[target[i]][2] = diffJK
        end

        --local gradInput = self.gradInput:index(1, torch.LongTensor(self.indices[target[i]])):add(self.gradK[target[i]][1] + self.gradK[target[i]][2])
        --self.gradInput:indexCopy(1, torch.LongTensor(self.indices[target[i]]), gradInput)
        print(i, self.gradK[target[i]][1]:sum(1),self.gradK[target[i]][2]:sum(1))
        self.gradInput[i] = self.gradInput[i] + diffX1X2[ind] --+ -self.gradK[target[i]][1]:sum(1)

        self.gradInput[j] = self.gradInput[j] - diffX1X2[ind] --+ -self.gradK[target[i]][2]:sum(1)
    end

    self.gradInput = torch.cmul(self.Li:expandAs(input), self.gradInput) / self.counter
    print(aa)
    --return aa
    return self.gradInput
end


function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput1(input, target)
    self.gradInput1 = torch.Tensor(input:size()):zero():type(torch.type(input))

    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
            if target[i] == target[j] and i ~= j then
                local subIJ = torch.csub(input[i], input[j])
                local normSubIJ = torch.norm(subIJ)
                for k = 1, input:size(1) do
                    if target[i] ~= target[k] then

                        local subIK = torch.csub(input[i], input[k])
                        local normSubIK = torch.norm(subIK)

                        local subJK = torch.csub(input[j], input[k])
                        local normSubJK = torch.norm(subJK)

                        local dividedIK = -torch.exp(self.alpha - normSubIK)
                        local dividedJK = -torch.exp(self.alpha - normSubJK)

                        local dividing = torch.exp(self.Li[i][1] - normSubIJ)
                        --self.gradInput1[i] = self.gradInput1[i] + (subIK * (dividedIK / (dividing * normSubIK)))
                        --self.gradInput1[k] = self.gradInput1[k] + -(subIK * (dividedIK / (dividing * normSubIK)))

                        --self.gradInput1[j] = self.gradInput1[j] + (subJK * (dividedJK / (dividing * normSubJK)))
                        --self.gradInput1[k] = self.gradInput1[k] + -(subJK * (dividedJK / (dividing * normSubJK)))
                    end
                end
                self.gradInput1[i] = self.gradInput1[i] + (subIJ / normSubIJ)

                self.gradInput1[j] = self.gradInput1[j] + (-subIJ / normSubIJ)
            end
        end
    end

    self.gradInput1 = torch.cmul(self.Li:expandAs(input), self.gradInput1) / self.counter

    return self.gradInput1
end
