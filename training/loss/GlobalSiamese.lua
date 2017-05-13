local SiamesePlusGlobalCriterion, parent = torch.class('nn.SiamesePlusGlobalCriterion', 'nn.Criterion')
local epsilon = 1e-35
function SiamesePlusGlobalCriterion:__init()
    parent.__init(self)
    self.alpha = 1
    self.Li = torch.Tensor()
    self.gradInput = {}
    self.t = 0.4
    self.teta = 1
    self.lambda = 1
end


function SiamesePlusGlobalCriterion:updateOutput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)

    self.dipos = (x1 - x2):norm(2, 2):cmul(target)
    self.dineg = (x1 - x2):norm(2, 2):cmul(1 - target)

    self.nupos = torch.sum(self.dipos, 2) / N
    self.nuneg = torch.sum(self.dineg, 2) / N

    self.sigmapos = (self.dipos - self.nupos):pow(2) / N
    self.sigmaneg = (self.dineg - self.nuneg):pow(2) / N

    self.J1g = (self.sigmapos + self.sigmaneg) + (self.lambda * torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(x1)), self.nupos - self.nuneg + self.t, 2), 2))


    self.Li = self.teta * ((x1 - x2):norm(2, 2):cmul(target)
            + torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(x1)), self.alpha - (x1 - x2):norm(2, 2), 2), 2):cmul(1 - target)) + self.J1g

    self.output = self.Li:sum() / N
    return self.output
end

function SiamesePlusGlobalCriterion:updateGradInput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)
    local norm = (x1 - x2):norm(2, 2):expandAs(x1 - x2) + epsilon
    local tar = torch.cmul(self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()), target:repeatTensor(x1:size(2), 1):t():type(x1:type())) + epsilon
    local nottar = torch.cmul(self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()), (target - 1):repeatTensor(x1:size(2), 1):t():type(x1:type())) + epsilon


    local a1 = (2 / N) * ((2 * self.dipos - self.nupos):expandAs(x1)
            - (1 / 2) * ((self.nuneg - self.nupos):expandAs(x1) - self.alpha):lt(0):type(torch.type(x1)))

    local a2 = (2 / N) * ((2 * self.dineg - self.nuneg):expandAs(x1)
            - (1 / 2) * ((self.nuneg - self.nupos):expandAs(x1) - self.alpha):lt(0):type(torch.type(x1)))
    self.gradInput[1] = ((torch.cdiv(x1 - x2, norm):cmul(tar)) / N
            + (torch.cdiv(x1 - x2, norm):cmul(nottar)) / N) + a1

    self.gradInput[2] = ((torch.cdiv(x2 - x1, norm):cmul(tar)) / N
            + (torch.cdiv(x2 - x1, norm):cmul(nottar)) / N) + a2

    return self.gradInput
end
