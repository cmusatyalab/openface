--DONE but not tested
local LMNNPushCriterion, parent = torch.class('nn.LMNNPushCriterion', 'nn.Criterion')

function LMNNPushCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function LMNNPushCriterion:updateOutput(inputs)
    local a = inputs[1] -- ancor
    local p = inputs[2] -- positive
    local n = inputs[3] -- negative
    local N = a:size(1)

    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), self.alpha + (a - p):norm(2, 2):pow(2) - (a - n):norm(2, 2):pow(2), 2), 2)
    self.output = self.Li:sum() / N
    return self.output
end

function LMNNPushCriterion:updateGradInput(inputs)
    local a = inputs[1] -- ancor
    local p = inputs[2] -- positive
    local n = inputs[3] -- negative
    local N = a:size(1)

    self.gradInput[1] = (n - p):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N)
    self.gradInput[2] = (p - a):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N)
    self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N)

    return self.gradInput
end
