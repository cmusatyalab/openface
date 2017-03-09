--DONE But Not Tested

optimLogger = optim.Logger(paths.concat(opt.save, 'diff.log'))
local DoubleMarginCriterion, parent = torch.class('nn.DoubleMarginCriterion', 'nn.Criterion')

function DoubleMarginCriterion:__init(alpha)
    parent.__init(self)
    self.alpha1 = 1
    self.alpha2 = alpha or 1
    self.Li = torch.Tensor()
end


function DoubleMarginCriterion:updateOutput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)
    optimLogger:add {
        (x1 - x2):norm(2, 2):pow(2):min(), (x1 - x2):norm(2, 2):pow(2):cmul(target):max(), (x1 - x2):norm(2, 2):min(),
        (x1 - x2):norm(2, 2):cmul(target):max()
    }
    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(x1)), (x1 - x2):norm(2, 2):pow(2) - self.alpha1, 2), 2):cmul(target)
            + torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(x1)), self.alpha2 - (x1 - x2):norm(2, 2):pow(2), 2), 2):cmul(1 - target)
    self.output = self.Li:sum() / N
    return self.output
end

function DoubleMarginCriterion:updateGradInput(inputs, target)
    local x1 = inputs[1]
    local x2 = inputs[2]
    local N = x1:size(1)

    self.gradInput = (4 * target - 2):repeatTensor(x1:size(2), 1):t():type(x1:type()):cmul(torch.cmul(x1 - x2, self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()))) / N

    return self.gradInput
end
