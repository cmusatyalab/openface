--DONE

local ImprovedTripletCriterion, parent = torch.class('nn.ImprovedTripletCriterion', 'nn.Criterion')

function ImprovedTripletCriterion:__init(alpha1, alpha2, beta)
    parent.__init(self)
    self.alpha2 = alpha2 or 0.01
    self.alpha1 = alpha1 or -1
    self.beta = beta or 0.002
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function ImprovedTripletCriterion:updateOutput(input)
    local a = input[1] -- ancor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)
    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), (a - p):norm(2, 2):pow(2) - (a - n):norm(2, 2):pow(2) + self.alpha2, 2), 2)
            + self.beta * torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)), (a - p):norm(2, 2):pow(2) + self.alpha1, 2), 2)
    self.output = self.Li:sum() / N
    return self.output
end

function ImprovedTripletCriterion:updateGradInput(input)
    local a = input[1] -- ancor
    local p = input[2] -- positive
    local n = input[3] -- negative
    local N = a:size(1)

    self.gradInput[1] = (2 * (n - p) + self.beta * (a - 2 * p)):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) / N)
    self.gradInput[2] = (2 * (p - a) + self.beta * (-2 * a + p)):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) / N)
    self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2), 1):t():type(a:type()) * 2 / N)

    return self.gradInput
end
