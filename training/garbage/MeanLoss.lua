--
-- Created by IntelliJ IDEA.
-- User: Cenk Bircanoglu
-- Date: 11/12/2016
-- Time: 16:22
-- To change this template use File | Settings | File Templates.
--

local MeanLossCriterion, parent = torch.class('nn.MeanLossCriterion', 'nn.Criterion')

function MeanLossCriterion:__init(alpha, numberPerClass)
    parent.__init(self)
    self.alpha = alpha or 0.2
    self.means = {}
    self.gradInput = {}
    self.Li = nil
    self.numberPerClass = numberPerClass or 3
    self.beta = 0.1
end


function MeanLossCriterion:updateOutput(inputs)
    local N = inputs:size(1)
    local idx = 1
    for i = 1, N / self.numberPerClass do
        self.means[i] = inputs[{ { idx, idx + self.numberPerClass - 1 }, {} }]:mean(1)
        idx = idx + self.numberPerClass
    end

    self.Li = torch.Tensor(N):fill(0)

    for i = 1, N do
        local classId1 = self:classId(i)
        local mean1 = self:findMean(classId1)
        local val1 = inputs[{ { i }, {} }]
        --self.Li[{ { i } }]:add( ((val1 - mean1):norm(2, 2):pow(2)))
        for j = 1, N do
            local classId2 = self:classId(j)

            if classId1 ~= classId2 then
                local val2 = inputs[{ { j }, {} }]
                local mean2 = self:findMean(classId2)
                self.Li[{ { i } }]:add(torch.max(torch.cat(torch.Tensor(val1:size(1)):zero():type(torch.type(val1)),
                    --self.beta * ((val1 - mean1):norm(2, 2):pow(2) + (val2 - mean2):norm(2, 2):pow(2)) +
                             self.alpha
                        - (val1 - val2):norm(2, 2):pow(2), 2)))
            end
        end
    end
    self.output = self.Li:sum() / N
    return self.output
end

function MeanLossCriterion:negativeSum(input, classId)
    local start = (classId - 1) * self.numberPerClass + 1
    local ends = start + self.numberPerClass - 1
    return input:sum(1) - input[{ { start, ends }, {} }]:sum(1)
end

function MeanLossCriterion:updateGradInput(inputs)
    local N = inputs:size(1)
    self.gradInput = torch.Tensor(inputs:size(1), inputs:size(2)):fill(0)
    for i = 1, N do
        local classId = self:classId(i)
        local mean = self:findMean(classId)
        local negSize = N - self.numberPerClass
        --self.gradInput[i] = (self:negativeSum(inputs, classId) - self.beta * (mean * negSize)):cmul(self.Li[{ { i } }]:gt(0):repeatTensor(inputs:size(2), 1):t():type(inputs[1]:type()) * 2 / N)
        self.gradInput[i] = (self:negativeSum(inputs, classId) / negSize
               -- - self.beta * (inputs[{ { i }, {} }]:cmul(mean))
            ):cmul(self.Li[{ { i } }]:gt(0):repeatTensor(inputs:size(2), 1):t():type(inputs[1]:type()) * 2 / N)
    end
    return self.gradInput
end



function MeanLossCriterion:findMean(id_)

    return self.means[id_]
end

function MeanLossCriterion:classId(i)
    local id_ = 1
    local ind = math.floor((i - 1) / self.numberPerClass)
    id_ = id_ + ind
    return id_
end
