local BatchKLDivCriterion, parent = torch.class('nn.BatchKLDivCriterion', 'nn.Criterion')
-- This function is only for handling the output of softmax layer

local epsilon = 1e-35 --to avoid calculating log(0) or dividing 0

function BatchKLDivCriterion:__init(margin)
    parent.__init(self)
    self.margin = margin or 2 -- 2 is the empirical value for mnist
    self.Li = nil
end


function BatchKLDivCriterion:updateOutput(inputs, target)

    local p = inputs[1]
    local q = inputs[2]

    local N = target:size(1)
    local KLDivQP = (torch.log(q + epsilon) - torch.log(p + epsilon)):cmul(q)

    self.Li = (torch.sum(KLDivQP, 2)):cmul(target) + torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(KLDivQP)), self.margin - torch.sum(KLDivQP, 2), 2), 2):cmul(1 - target)

    return self.Li:sum() * 2 / N
end

function BatchKLDivCriterion:updateGradInput(inputs, target)

    local p = inputs[1]
    local q = inputs[2]
    local N = target:size(1)

    local pdivq = q:cdiv(p + epsilon):mul(-1)
    local li = torch.cmul(pdivq, self.Li:gt(0):repeatTensor(p:size(2), 1):t():type(p:type()))
    self.gradInput = torch.cmul(li, target:repeatTensor(p:size(2), 1):t():type(p:type())) + torch.cmul(li, (target - 1):repeatTensor(p:size(2), 1):t():type(p:type()))
    return self.gradInput
end
