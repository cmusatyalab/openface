--
-- Created by IntelliJ IDEA.
-- User: Cenk Bircanoglu
-- Date: 11/12/2016
-- Time: 16:22
-- To change this template use File | Settings | File Templates.
-- (x1 -x2)^2 / (x1+x2)^2 minimize x1 and x2 pairs from different classes

local MinDiffLossCriterion, parent = torch.class('nn.MinDiffLossCriterion', 'nn.Criterion')

function MinDiffLossCriterion:__init(alpha, numberPerClass)
    parent.__init(self)
    self.alpha = alpha or 0.01
    self.means = {}
    self.gradInput = {}
    self.Li = nil
    self.numberPerClass = numberPerClass or 3
end


function MinDiffLossCriterion:updateOutput(inputs)
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
        for j = 1, N do
            local classId2 = self:classId(j)

            if classId1 ~= classId2 then
                local val2 = inputs[{ { j }, {} }]
                local mean2 = self:findMean(classId2)
                self.Li[{ { i } }]:add(torch.max(torch.cat(torch.Tensor(val1:size(1)):zero():type(torch.type(val1)),
                    (((val1 - val2):norm(2, 2):pow(2)):cdiv((val1 + val2):norm(2, 2):pow(2)) + self.alpha * (val1 - mean1):norm(2, 2):pow(2)), 2)))
            end
        end
    end
    self.output = self.Li:sum() / N
    return self.output
end

function MinDiffLossCriterion:negativeSum(input, classId)
    local start = (classId - 1) * self.numberPerClass + 1
    local ends = start + self.numberPerClass - 1
    return input:sum(1) - input[{ { start, ends }, {} }]:sum(1)
end

function MinDiffLossCriterion:updateGradInput(inputs)
    local N = inputs:size(1)
    self.gradInput = torch.Tensor(inputs:size(1), inputs:size(2)):fill(0)
    for i = 1, N do
        local classId1 = self:classId(i)
        local mean1 = self:findMean(classId1)
        local val1 = inputs[{ { i }, {} }]
        for j = 1, N do
            local classId2 = self:classId(j)
            if classId1 ~= classId2 then
                local val2 = inputs[{ { j }, {} }]
                self.gradInput[i]:add((((4 * val2):cdiv((val1 + val2):pow(2))) + self.alpha * ((val1 - mean1))
                        - ((8 * torch.pow(val2, 2)):cdiv((val1 + val2):pow(3)))):cmul(self.Li[{ { i } }]:gt(0):repeatTensor(inputs:size(2), 1):t():type(inputs[1]:type()) * 2 / N))
            end
        end
    end
    return self.gradInput
end



function MinDiffLossCriterion:findMean(id_)

    return self.means[id_]
end

function MinDiffLossCriterion:classId(i)
    local id_ = 1
    local ind = math.floor((i - 1) / self.numberPerClass)
    id_ = id_ + ind
    return id_
end
