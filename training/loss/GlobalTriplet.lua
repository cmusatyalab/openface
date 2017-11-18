local TripletPlusGlobalCriterion, parent = torch.class('nn.TripletPlusGlobalCriterion', 'nn.Criterion')
epsilon = 1e-35
function TripletPlusGlobalCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 0.01
    self.Li = torch.Tensor()
    self.gradInput = {}
    self.t = 0.4
    self.teta = 1
    self.lambda = 0.8
end

function TripletPlusGlobalCriterion:updateOutput(input)
    local a = input[1] -- ancor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)
    local apnorm = (a - p):norm(2, 2)
    local annorm = (a - n):norm(2, 2)
    self.dipos = torch.pow(apnorm, 2) / 4
    self.dineg = torch.pow(annorm, 2) / 4

    self.nupos = torch.sum(self.dipos, 2) / N
    self.nuneg = torch.sum(self.dineg, 2) / N

    self.sigmapos = (self.dipos - self.nupos):pow(2) / N
    self.sigmaneg = (self.dineg - self.nuneg):pow(2) / N

    self.J1g = (self.sigmapos + self.sigmaneg) + (self.lambda * torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), self.nupos - self.nuneg + self.t, 2), 2))

    self.Li = self.teta * torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), (a - p):norm(2, 2):pow(2) - (a - n):norm(2, 2):pow(2) + self.alpha, 2), 2) + self.J1g
    self.output = self.Li:sum() / N
    return self.output
end

function TripletPlusGlobalCriterion:updateGradInput(input)
    local a = input[1] -- ancor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)

    local x1 = -(1 / 2 * N) * (2 * (torch.cmul((self.dipos - self.nupos):expandAs(p), p) + torch.cmul((self.dineg - self.nuneg):expandAs(n), n)) +
            (self.lambda * torch.cmul((p - n), ((self.nuneg - self.nupos):expandAs(p) - self.t):lt(0):type(torch.type(a)))))

    local x2 = -(1 / 2 * N) * (2 * (torch.cmul((self.dipos - self.nupos):expandAs(a), a)) +
            (torch.cmul(a, ((self.nuneg - self.nupos):expandAs(p) - self.t):lt(0):type(torch.type(a)))))

    local x3 = -(1 / 2 * N) * (2 * (torch.cmul((self.dineg - self.nuneg):expandAs(a), a)) -
            (torch.cmul(a, ((self.nuneg - self.nupos):expandAs(p) - self.t):lt(0):type(torch.type(a)))))

    self.gradInput[1] = (n - p):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N) + x1
    self.gradInput[2] = (p - a):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N) + x2
    self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N) + x3
    return self.gradInput
end
