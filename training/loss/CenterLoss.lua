--
-- Created by IntelliJ IDEA.
-- User: Cenk Bircanoglu
-- Date: 11/12/2016
-- Time: 16:22
-- To change this template use File | Settings | File Templates.
--

local CenterLossCriterion, parent = torch.class('nn.CenterLossCriterion', 'nn.Criterion')

function CenterLossCriterion:__init(alpha, numberPerClass)
    parent.__init(self)
    self.alpha = alpha * 0.01
    self.means = {}
    self.gradInput = {}
    self.Li = nil
    self.numberPerClass = numberPerClass or 3
end


function CenterLossCriterion:updateOutput(inputs)
    local N = inputs:size(1)
    local idx = 1
    for i = 1, N / self.numberPerClass do
        self.means[i] = inputs[{ { idx, idx + self.numberPerClass - 1 }, {} }]:mean(1)
        idx = idx + self.numberPerClass
    end
    print(self.means)
    self.Li = torch.Tensor(N * N):fill(0)

    for i = 1, N do
        local classId1 = self:classId(i)
        local mean1 = self:findMean(classId1)
        local val1 = inputs[{ { i }, {} }]
        for j = 1, #self.means do
            if classId1 ~= j then
                self.Li[{ { i } }]:add(self.alpha * self.numberPerClass * (val1:norm(2, 2):pow(2):cdiv((mean1 - self.means[j]):norm(2, 2):pow(2))))
            end
        end
    end
    print(self.Li)
    self.output = self.Li:sum() / N
    return self.output
end

function CenterLossCriterion:updateGradInput(inputs)
    local N = inputs:size(1)
    self.gradInput = torch.Tensor(inputs:size(1), inputs:size(2)):fill(0)
    for i = 1, N do
        local classId = self:classId(i)
        local mean = self:findMean(classId)
        self.gradInput[i] = (inputs[{ { i }, {} }] - mean):cmul(self.Li[{ { i } }]:gt(0):repeatTensor(inputs:size(2), 1):t():type(inputs[1]:type()) * 2 / N)
    end
    return self.gradInput
end



function CenterLossCriterion:findMean(id_)

    return self.means[id_]
end

function CenterLossCriterion:classId(i)
    local id_ = 1
    local ind = math.floor((i - 1) / self.numberPerClass)
    id_ = id_ + ind
    return id_
end
