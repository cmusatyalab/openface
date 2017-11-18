--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 12/12/2016
-- Time: 22:54
-- To change this template use File | Settings | File Templates.
--


local ConsLossCriterion, parent = torch.class('nn.ConsLossCriterion', 'nn.Criterion')

function ConsLossCriterion:__init(alpha, numberPerClass)
    parent.__init(self)
    self.alpha = 0.1e-8
    self.means = {}
    self.gradInput = {}
    self.Li = nil
    self.numberPerClass = numberPerClass or 3
end


function ConsLossCriterion:updateOutput(inputs)
    local N = inputs:size(1)

    self.Li = torch.Tensor(N, N):fill(0)
    for i = 1, N do
        local classId1 = self:classId(i)
        local val1 = inputs[{ { i }, {} }]
        for j = 1, N do
            local classId2 = self:classId(j)
            if classId1 ~= classId2 then
                local val2 = inputs[{ { j }, {} }]
                self.Li[{ { i }, { j } }] = (val1 - val2):norm(2, 2):pow(2)
            end
        end
    end
    self.output = self.Li:sum() / N
    return self.output
end

function ConsLossCriterion:updateGradInput(inputs)
    local N = inputs:size(1)
    self.gradInput = torch.Tensor(inputs:size(1), inputs:size(2)):fill(0)
    for i = 1, N do
        local classId1 = self:classId(i)
        local val1 = inputs[{ { i }, {} }]
        for j = 1, N do
            local classId2 = self:classId(j)
            if classId1 ~= classId2 then
                local val2 = inputs[{ { j }, {} }]
                self.gradInput[i]:add((val1 - val2):cmul(self.Li[{ { i }, { j } }]:gt(0):repeatTensor(val1:size(2), 1):t():type(inputs[1]:type()) * 2 / N))
                self.gradInput[j]:add((val1 - val2):cmul(self.Li[{ { i }, { j } }]:gt(0):repeatTensor(val1:size(2), 1):t():type(inputs[1]:type()) * 2  / N))

            end
        end
    end
    return self.gradInput
end

function ConsLossCriterion:classId(i)
    local id_ = 1
    local ind = math.floor((i - 1) / self.numberPerClass)
    id_ = id_ + ind
    return id_
end
