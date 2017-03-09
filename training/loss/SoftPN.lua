-- Taken from Elad Hoffer's TripletNet https://github.com/eladhoffer
-- Hinge loss ranking could also be used, see below
-- https://github.com/torch/nn/blob/master/doc/criterion.md#nn.MarginRankingCriterion

local SoftPNCriterion, parent = torch.class('nn.SoftPNCriterion', 'nn.Criterion')

function SoftPNCriterion:__init()
    parent.__init(self)
    self.SoftMax = nn.SoftMax()
    self.MSE = nn.MSECriterion()
    self.Target = torch.Tensor()
end

function SoftPNCriterion:createTarget(input, target)
    local target = target or 1
    self.Target:resizeAs(input):typeAs(input):zero()
    self.Target[{ {}, target }]:add(1)
end

function SoftPNCriterion:updateOutput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end
    self.output = self.MSE:updateOutput(self.SoftMax:updateOutput(input), self.Target)
    return self.output
end

function SoftPNCriterion:updateGradInput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end

    self.gradInput = self.SoftMax:updateGradInput(input, self.MSE:updateGradInput(self.SoftMax.output, self.Target))
    return self.gradInput
end

function SoftPNCriterion:type(t)
    parent.type(self, t)
    self.SoftMax:type(t)
    self.MSE:type(t)
    self.Target = self.Target:type(t)
    return self
end