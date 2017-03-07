--DONE but not tested
local LargeMarginNearestNeighborCriterion, parent = torch.class('nn.LargeMarginNearestNeighborCriterion', 'nn.Criterion')

function LargeMarginNearestNeighborCriterion:__init(alpha, beta)
    parent.__init(self)
    self.alpha = alpha or 1
    self.beta = beta or 0.5
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function LargeMarginNearestNeighborCriterion:updateOutput(inputs, target)
    local a = inputs[1] -- ancor
    local p = inputs[2] -- positive
    local n = inputs[3] -- negative
    local N = a:size(1)
    self.Li = self.beta * torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), self.alpha + (a - p):norm(2, 2):pow(2) - (a - n):norm(2, 2):pow(2), 2), 2):cmul(1 - target)
            + (1 - self.beta) * (a - p):norm(2, 2):pow(2)
    self.output = self.Li:sum() / N
    return self.output
end

function LargeMarginNearestNeighborCriterion:updateGradInput(input)
    local a = input[1] -- ancor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)
    local li = self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type())

    self.gradInput[1] = (n - p):cmul(li * 2 * self.beta / N) + (a - 2 * p):cmul(li * 2 * (1 - self.beta) / N)
    self.gradInput[2] = (p - a):cmul(li * 2 * self.beta / N) + (p - 2 * a):cmul(li * 2 * (1 - self.beta) / N)
    self.gradInput[3] = (a - n):cmul(li * 2 * self.beta / N)

    return self.gradInput
end
