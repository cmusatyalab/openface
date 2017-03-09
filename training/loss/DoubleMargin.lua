--DONE But Not Tested


local DoubleMarginCriterion, parent = torch.class('nn.DoubleMarginCriterion', 'nn.Criterion')

function DoubleMarginCriterion:__init(alpha)
    parent.__init(self)
    self.alpha1 = 1
    self.alpha2 = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end


function DoubleMarginCriterion:updateOutput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)

    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(x1)), (x1 - x2):norm(2, 2) - self.alpha1, 2), 2):cmul(target)
            + torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(x1)), self.alpha2 - (x1 - x2):norm(2, 2), 2), 2):cmul(1 - target)
    self.output = self.Li:sum() / N
    return self.output
end

function DoubleMarginCriterion:updateGradInput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)
    local norm = (x1 - x2):norm(2, 2):expandAs(x1 - x2)
    local tar = torch.cmul(self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()), target:repeatTensor(x1:size(2), 1):t():type(x1:type()))
    local nottar = torch.cmul(self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()), (target - 1):repeatTensor(x1:size(2), 1):t():type(x1:type()))

    self.gradInput[1] = (torch.cdiv(x1 - x2, norm):cmul(tar)) / N
            + (torch.cdiv(x1 - x2, norm):cmul(nottar)) / N

    self.gradInput[2] = (torch.cdiv(x2 - x1, norm):cmul(tar)) / N
            + (torch.cdiv(x2 - x1, norm):cmul(nottar)) / N

    return self.gradInput
end
