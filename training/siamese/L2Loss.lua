local L2LossCriterion, parent = torch.class('nn.L2LossCriterion', 'nn.Criterion')

function L2LossCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function L2LossCriterion:updateOutput(output, labels)
    local x1 = output[1] -- ancor
    local x2 = output[2] -- positive
    local N = x1:size(1)
    local fixedLabels = labels:clone()
    fixedLabels:apply(function(label)
        if label == 1 then
            return 0
        elseif label == -1 then
            return 1
        end
    end)
    --This one is true and tested
    local diff_norm = (x1 - x2):norm(2, 2)
    self.Li = torch.cmul((torch.ones(fixedLabels:size(1)) - fixedLabels), (diff_norm:pow(2))) / 2 +
            torch.cmul(fixedLabels, (torch.max(torch.zeros(fixedLabels:size(1)), self.alpha - diff_norm, 2) / 2):pow(2))

    self.output = self.Li:sum() / N
    return self.output
end

function L2LossCriterion:updateGradInput(output, labels)
    local x1 = output[1] -- ancor
    local x2 = output[2] -- positive
    local fixedLabels = labels:clone()
    local N = x1:size(1)
    fixedLabels:apply(function(label)
        if label == 1 then
            return 0
        elseif label == -1 then
            return 1
        end
    end)
    -- TODO
    local diff = (x1 - x2)
    self.gradInput[1] = ((1 - fixedLabels):repeatTensor(x1:size(1), 1):t() * diff +
            fixedLabels:repeatTensor(x1:size(1), 1):t() * (diff - (self.alpha * diff / (diff:norm(2)))) / 2):cmul(self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()) / 2 * N)
    self.gradInput[2] = ((1 - fixedLabels):repeatTensor(x1:size(1), 1):t() * -diff +
            fixedLabels:repeatTensor(x1:size(1), 1):t() * (-diff - (self.alpha * -diff / (diff:norm(2)))) / 2):cmul(self.Li:gt(0):repeatTensor(x1:size(2), 1):t():type(x1:type()) / 2 * N)

    return self.gradInput
end
