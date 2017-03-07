--DONE But Not Tested

local DoubleMarginCriterion, parent = torch.class('nn.DoubleMarginCriterion', 'nn.Criterion')

function DoubleMarginCriterion:__init(alpha1, alpha2)
    parent.__init(self)
    self.alpha1 = alpha1 or 1
    self.alpha2 = alpha2 or 2
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function DoubleMarginCriterion:updateOutput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)
    local norm = (x1 - x2):norm(2, 2):pow(2)
    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(norm)), norm - self.alpha1, 2), 2):cmul(target)
            + torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(norm)), self.alpha2 - norm, 2), 2):cmul(1 - target)
    self.output = self.Li:sum() / N
    return self.output
end

function DoubleMarginCriterion:updateGradInput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)


    local li = self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type())
    local diff = x1 - x2
    self.gradInput[1] = (4 * target - 2):repeatTensor(x1:size(2), 1):t():type(x1:type()):cmul(torch.cmul(diff, li)) / N
    self.gradInput[2] = (4 * target - 2):repeatTensor(x1:size(2), 1):t():type(x1:type()):cmul(torch.cmul(-diff, li)) / N
    return self.gradInput
end
