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
    local i = 0
    self.Li = labels:clone()
    self.Li:apply(function(label)
        i = i + 1
        if label == 1 then
            return (math.max(0, self.alpha - (x1[i] - x2[i]):norm(2)) / 2) ^ 2
        elseif label == -1 then
            return ((x1[i] - x2[i]):norm(2) ^ 2) / 2
        end
    end)
    self.output = self.Li:sum() / N
    return self.output
end

function L2LossCriterion:updateGradInput(output, labels)
    local x1 = output[1] -- ancor
    local x2 = output[2] -- positive
    local N = x1:size(1)
    self.gradInput[1] = torch.Tensor(x1:size(1), x1:size(2))
    self.gradInput[2] = torch.Tensor(x1:size(1), x1:size(2))

    function apply_to_slices(tensor, dimension, func)
        for i, slice in ipairs(tensor:split(1, dimension)) do
            func(slice, i)
        end
        return tensor
    end

    apply_to_slices(self.gradInput[1], 1, function(slice, i)
        if labels[i] == 1 then
            self.gradInput[1][i] = (x1[i] - x2[i]):csub(self.alpha):mul(self.Li[i] / 2 * N)
        elseif labels[i] == -1 then
            self.gradInput[1][i] = (x1[i] - x2[i]):mul(self.Li[i] / 2 * N)
        end
    end)

    apply_to_slices(self.gradInput[2], 1, function(slice, i)
        if labels[i] == 1 then
            self.gradInput[2][i] = (x2[i] - x1[i]):add(self.alpha):mul(self.Li[i] / 2 * N)
        elseif labels[i] == -1 then
            self.gradInput[2][i] = (x2[i] - x1[i]):mul(self.Li[i] / 2 * N)
        end
    end)
    return self.gradInput
end
